import numpy as np
from scipy.linalg import solve, LinAlgError

class KalmanTracker:
    def __init__(self, initial_pos, max_velocity=30.0):
        # state: [x, y, vx, vy, ax, ay]
        self.x = np.array([[initial_pos[0]], [initial_pos[1]], [0], [0], [0], [0]], dtype=np.float64)
        self.max_velocity = max_velocity
        self.frames_without_update = 0  # occlusion tracking counter
        
        # Track historical positions for trajectory validation
        self.position_history = []
        self.max_history_length = 30
        self.last_confirmed_position = initial_pos
        self.last_confirmed_time = 0
        
        # Track exit zones and impossible reappearance regions
        self.exit_velocity = None
        self.exit_position = None
        self.exit_time = None
        
        # Scene boundary awareness (can be set externally if known)
        self.scene_bounds = None  # [min_x, min_y, max_x, max_y]
        
        # Adaptive process noise with separate modes for stationary/moving
        self.base_Q_moving = np.diag([0.5, 0.5, 2.0, 2.0, 1.0, 1.0])
        self.base_Q_stationary = np.diag([0.2, 0.2, 0.5, 0.5, 0.3, 0.3])
        self.base_Q = self.base_Q_moving.copy()
        self.is_moving_state = False
        self.motion_transition_count = 0
        
        # Track confidence and consistency
        self.tracking_confidence = 1.0
        self.consistency_score = 1.0
        self.appearance_features = None  # Can store appearance descriptors if available
        
        # state transition matrix: x_k = F x_{k-1} + w_k
        self.F = np.array([[1, 0, 1, 0, 0.5, 0],
                          [0, 1, 0, 1, 0, 0.5],
                          [0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=np.float64)
        
        # measurement matrix: z_k = H x_k + v_k
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]], dtype=np.float64)
        
        # process covariance: Q = E[w w^T]
        self.Q = self.base_Q.copy()
        # measurement covariance: R = E[v v^T]
        self.R = np.eye(2) * 1.0
        # error covariance: P = E[(x - x̂)(x - x̂)^T]
        self.P = np.diag([10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        
        # smoothing parameters
        self.occlusion_smoothing = 0.9
        self.velocity_smoothing = 0.8
        self.motion_transition_threshold = 5  # frames to confirm motion state change
        
        # Teleportation detection parameters
        self.teleportation_threshold = 150.0  # max reasonable distance jump
        self.reappearance_velocity_threshold = 5.0  # max initial velocity for new appearance
     
    def predict(self):
        """
        Prediction step: x̂_k|k-1 = F x̂_k-1|k-1, P_k|k-1 = F P_k-1|k-1 F^T + Q
        """
        # Store previous position before prediction
        prev_pos = (self.x[0, 0], self.x[1, 0])
        
        # Project state and covariance forward
        self.x = self.F @ self.x
        
        # Adaptive noise: increase uncertainty during occlusion with exponential smoothing
        occlusion_factor = 1.0 + (1 - self.occlusion_smoothing) * self.frames_without_update
        self.Q = self.base_Q * min(occlusion_factor, 5.0)
        
        # Decay confidence during occlusion
        self.tracking_confidence *= (0.95 if self.frames_without_update > 0 else 1.0)
        self.tracking_confidence = max(0.1, self.tracking_confidence)
        
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Check if predicted position is going out of bounds
        if self.scene_bounds is not None:
            min_x, min_y, max_x, max_y = self.scene_bounds
            curr_x, curr_y = self.x[0, 0], self.x[1, 0]
            
            # If heading out of bounds, store exit information
            if (curr_x < min_x or curr_x > max_x or curr_y < min_y or curr_y > max_y):
                if self.exit_position is None:
                    self.exit_position = prev_pos
                    self.exit_velocity = (self.x[2, 0], self.x[3, 0])
                    self.exit_time = self.frames_without_update
        
        # Detect motion state changes
        current_velocity = np.sqrt(self.x[2, 0]**2 + self.x[3, 0]**2)
        new_moving_state = current_velocity > 2.0  # threshold for considering movement
        
        if new_moving_state != self.is_moving_state:
            self.motion_transition_count += 1
            if self.motion_transition_count >= self.motion_transition_threshold:
                self.is_moving_state = new_moving_state
                self.base_Q = self.base_Q_moving if self.is_moving_state else self.base_Q_stationary
                self.motion_transition_count = 0
        else:
            self.motion_transition_count = max(0, self.motion_transition_count - 1)
        
        # Velocity constraint with momentum preservation
        if current_velocity > self.max_velocity:
            scale = self.max_velocity / current_velocity
            # Smooth velocity reduction to maintain direction
            self.x[2, 0] *= self.velocity_smoothing * scale + (1 - self.velocity_smoothing)
            self.x[3, 0] *= self.velocity_smoothing * scale + (1 - self.velocity_smoothing)
        
        # Increment occlusion counter
        self.frames_without_update += 1
        
        # Update position history
        self._update_history((int(self.x[0, 0]), int(self.x[1, 0])))
        
        return (int(self.x[0, 0]), int(self.x[1, 0]))
    
    def update(self, measurement, max_distance=80.0):
        """
        Update step: K = P H^T (H P H^T + R)^{-1}, x̂ = x̂ + K(z - H x̂), P = (I - K H) P
        Enhanced with spatial awareness and physical plausibility checks
        """
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
        
        # Check for physically impossible reappearance
        if not self._is_physically_plausible_update(measurement):
            return False
        
        # Teleportation detection
        if self.frames_without_update <= 2:
            distance_jump = np.linalg.norm(self.x[:2] - z)
            expected_distance = np.linalg.norm(self.x[2:4]) * (1 + self.frames_without_update)
            
            # Check if this is an unreasonable jump
            if distance_jump > self.teleportation_threshold and distance_jump > 3 * expected_distance:
                # This might be a different object
                return False
        
        # Mahalanobis distance gating with improved numerical stability
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        
        try:
            # Use solve instead of explicit inversion for numerical stability
            mahal_dist_squared = float(y.T @ solve(S, y, assume_a='pos'))
            mahal_dist = np.sqrt(mahal_dist_squared)
            
            # Adaptive threshold based on motion state, occlusion, and confidence
            base_threshold = 3.0  # 3-sigma
            if not self.is_moving_state:
                base_threshold *= 0.7  # tighter gate when stationary
            
            # Adjust threshold based on tracking confidence
            confidence_factor = 2.0 - self.tracking_confidence  # ranges from 1.0 to 1.9
            threshold = base_threshold * (1.0 + 0.1 * self.frames_without_update) * confidence_factor
            
            if mahal_dist > threshold:
                return False
        except (np.linalg.LinAlgError, LinAlgError):
            # Fallback to Euclidean distance with adaptive threshold
            distance = np.linalg.norm(self.x[:2] - z)
            adaptive_threshold = max_distance * (1.0 + 0.1 * self.frames_without_update) * (2.0 - self.tracking_confidence)
            if distance > adaptive_threshold:
                return False
        
        # Directional consistency check with velocity validation
        if self.frames_without_update <= 2 and np.linalg.norm(self.x[2:4]) > 5.0:
            predicted_movement = self.x[2:4] * (1 + self.frames_without_update)
            actual_movement = z - self.x[:2]
            
            # Only check direction if both movements are significant
            if np.linalg.norm(predicted_movement) > 3.0 and np.linalg.norm(actual_movement) > 3.0:
                dot_product = (predicted_movement.T @ actual_movement) / (
                    np.linalg.norm(predicted_movement) * np.linalg.norm(actual_movement))
                
                if dot_product < -0.5:  # Reject strong opposite movements
                    return False
        
        # Check for sudden appearance with high velocity (unlikely for new objects)
        if self.frames_without_update > 20:  # Long occlusion
            implied_velocity = np.linalg.norm(z - self.x[:2]) / max(1, self.frames_without_update)
            if implied_velocity > self.reappearance_velocity_threshold:
                # This suggests it's a new object, not the tracked one
                return False
        
        # Kalman gain computation with improved numerical stability
        try:
            # Use solve for better numerical properties than explicit inversion
            PHT = self.P @ self.H.T
            K = solve(S, PHT.T, assume_a='pos').T
        except (np.linalg.LinAlgError, LinAlgError):
            # Regularized inversion fallback
            S_reg = S + np.eye(2) * 1e-6
            K = self.P @ self.H.T @ np.linalg.inv(S_reg)
        
        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # Ensure symmetry of covariance matrix
        self.P = (self.P + self.P.T) / 2.0
        
        # Update tracking metrics
        self._update_consistency_score(measurement)
        self.tracking_confidence = min(1.0, self.tracking_confidence + 0.1)
        
        # Detect if this update suggests a motion state change
        post_update_velocity = np.sqrt(self.x[2, 0]**2 + self.x[3, 0]**2)
        if (post_update_velocity > 3.0) != self.is_moving_state:
            self.motion_transition_count += 1
        else:
            self.motion_transition_count = max(0, self.motion_transition_count - 1)
        
        # Update confirmed position
        self.last_confirmed_position = measurement
        self.last_confirmed_time = 0
        
        # Reset occlusion counter and exit information
        self.frames_without_update = 0
        self.exit_position = None
        self.exit_velocity = None
        self.exit_time = None
        
        return True
    
    def _is_physically_plausible_update(self, measurement):
        """
        Check if an update is physically plausible given the tracking history
        """
        # If we have exit information, check if reappearance makes sense
        if self.exit_position is not None and self.exit_velocity is not None:
            time_since_exit = self.frames_without_update - self.exit_time
            
            # Calculate where the object could have traveled
            max_travel_distance = self.max_velocity * time_since_exit
            
            # If exited to the right and reappearing on the left (or vice versa)
            exit_x, exit_y = self.exit_position
            new_x, new_y = measurement
            
            # Check for impossible wraparound
            if self.scene_bounds is not None:
                min_x, min_y, max_x, max_y = self.scene_bounds
                
                # Exited right, appearing left
                if exit_x > max_x * 0.9 and new_x < min_x + (max_x - min_x) * 0.1:
                    if self.exit_velocity[0] > 0:  # Was moving right
                        return False
                
                # Exited left, appearing right
                if exit_x < min_x + (max_x - min_x) * 0.1 and new_x > max_x * 0.9:
                    if self.exit_velocity[0] < 0:  # Was moving left
                        return False
                
                # Similar checks for vertical
                if exit_y > max_y * 0.9 and new_y < min_y + (max_y - min_y) * 0.1:
                    if self.exit_velocity[1] > 0:  # Was moving down
                        return False
                
                if exit_y < min_y + (max_y - min_y) * 0.1 and new_y > max_y * 0.9:
                    if self.exit_velocity[1] < 0:  # Was moving up
                        return False
            
            # Check if distance is plausible given time elapsed
            distance_from_exit = np.sqrt((new_x - exit_x)**2 + (new_y - exit_y)**2)
            if distance_from_exit > max_travel_distance * 1.5:  # Allow some margin
                return False
        
        return True
    
    def _update_history(self, position):
        """
        Update position history for trajectory analysis
        """
        self.position_history.append(position)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
    
    def _update_consistency_score(self, measurement):
        """
        Update consistency score based on measurement alignment with predictions
        """
        predicted_pos = (self.x[0, 0], self.x[1, 0])
        error = np.sqrt((measurement[0] - predicted_pos[0])**2 + 
                       (measurement[1] - predicted_pos[1])**2)
        
        # Normalize error to a score between 0 and 1
        normalized_error = min(1.0, error / self.max_velocity)
        new_score = 1.0 - normalized_error
        
        # Exponential moving average
        alpha = 0.3
        self.consistency_score = alpha * new_score + (1 - alpha) * self.consistency_score
    
    def set_scene_bounds(self, bounds):
        """
        Set scene boundaries for spatial awareness
        bounds: [min_x, min_y, max_x, max_y]
        """
        self.scene_bounds = bounds
    
    def get_velocity(self):
        return (self.x[2, 0], self.x[3, 0])
    
    def get_acceleration(self):
        return (self.x[4, 0], self.x[5, 0])

    def validate_kalman_prediction(self, max_acceleration=20.0):
        """
        Validate physical plausibility: ‖v‖ ≤ v_max, ‖a‖ ≤ a_max
        """
        vel_mag = np.linalg.norm(self.x[2:4])
        acc_mag = np.linalg.norm(self.x[4:6])
        return vel_mag <= self.max_velocity and acc_mag <= max_acceleration
    
    def is_occluded(self, max_frames=10):
        """
        Check occlusion status: t_since_update > threshold
        """
        return self.frames_without_update > max_frames
    
    def is_moving(self):
        """
        Check if target is currently moving
        """
        return self.is_moving_state and np.linalg.norm(self.x[2:4]) > 2.0
    
    def get_tracking_confidence(self):
        """
        Get current tracking confidence (0-1)
        """
        return self.tracking_confidence
    
    def get_consistency_score(self):
        """
        Get trajectory consistency score (0-1)
        """
        return self.consistency_score