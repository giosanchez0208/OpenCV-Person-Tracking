import numpy as np

class KalmanTracker:
    def __init__(self, initial_pos, max_velocity=30.0):
        # state: [x, y, vx, vy, ax, ay]
        self.x = np.array([[initial_pos[0]], [initial_pos[1]], [0], [0], [0], [0]], dtype=float)
        self.max_velocity = max_velocity
        self.frames_without_update = 0  # occlusion tracking counter
        
        # Adaptive process noise with separate modes for stationary/moving
        self.base_Q_moving = np.diag([0.5, 0.5, 2.0, 2.0, 1.0, 1.0])
        self.base_Q_stationary = np.diag([0.2, 0.2, 0.5, 0.5, 0.3, 0.3])
        self.base_Q = self.base_Q_moving.copy()
        self.is_moving = False
        self.motion_transition_count = 0
        
        # state transition matrix: x_k = F x_{k-1} + w_k
        self.F = np.array([[1, 0, 1, 0, 0.5, 0],
                          [0, 1, 0, 1, 0, 0.5],
                          [0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)
        
        # measurement matrix: z_k = H x_k + v_k
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]], dtype=float)
        
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
     
    def predict(self):
        """
        Prediction step: x̂_k|k-1 = F x̂_k-1|k-1, P_k|k-1 = F P_k-1|k-1 F^T + Q
        """
        # Project state and covariance forward
        self.x = self.F @ self.x
        
        # Adaptive noise: increase uncertainty during occlusion with exponential smoothing
        occlusion_factor = 1.0 + (1 - self.occlusion_smoothing) * self.frames_without_update
        self.Q = self.base_Q * min(occlusion_factor, 5.0)
        
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Detect motion state changes
        current_velocity = np.sqrt(self.x[2, 0]**2 + self.x[3, 0]**2)
        new_moving_state = current_velocity > 2.0  # threshold for considering movement
        
        if new_moving_state != self.is_moving:
            self.motion_transition_count += 1
            if self.motion_transition_count >= self.motion_transition_threshold:
                self.is_moving = new_moving_state
                self.base_Q = self.base_Q_moving if self.is_moving else self.base_Q_stationary
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
        
        return (int(self.x[0, 0]), int(self.x[1, 0]))
    
    def update(self, measurement, max_distance=80.0):
        """
        Update step: K = P H^T (H P H^T + R)^{-1}, x̂ = x̂ + K(z - H x̂), P = (I - K H) P
        """
        # Mahalanobis distance gating with Cholesky decomposition for stability
        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        
        try:
            # Use Cholesky decomposition for numerical stability
            L = np.linalg.cholesky(S)
            Linv = np.linalg.inv(L)
            Sinv = Linv.T @ Linv
            mahal_dist = np.sqrt(y.T @ Sinv @ y)[0, 0]
            
            # Adaptive threshold based on motion state and occlusion
            base_threshold = 3.0  # 3-sigma
            if not self.is_moving:
                base_threshold *= 0.7  # tighter gate when stationary
            threshold = base_threshold * (1.0 + 0.1 * self.frames_without_update)
            
            if mahal_dist > threshold:
                return False
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance with adaptive threshold
            distance = np.linalg.norm(self.x[:2] - z)
            adaptive_threshold = max_distance * (1.0 + 0.1 * self.frames_without_update)
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
        
        # Kalman gain computation with regularization
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Regularized inversion fallback
            S_reg = S + np.eye(2) * 1e-5
            K = self.P @ self.H.T @ np.linalg.inv(S_reg)
        
        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        # Ensure symmetry of covariance matrix
        self.P = (self.P + self.P.T) / 2.0
        
        # Detect if this update suggests a motion state change
        post_update_velocity = np.sqrt(self.x[2, 0]**2 + self.x[3, 0]**2)
        if (post_update_velocity > 3.0) != self.is_moving:
            self.motion_transition_count += 1
        else:
            self.motion_transition_count = max(0, self.motion_transition_count - 1)
        
        # Reset occlusion counter
        self.frames_without_update = 0
        return True
        
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
        return self.is_moving and np.linalg.norm(self.x[2:4]) > 2.0