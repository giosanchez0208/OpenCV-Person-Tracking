import numpy as np

class KalmanTracker:
    def __init__(self, initial_pos):
        # state: [x, y, vx, vy, ax, ay]
        self.x = np.array([[initial_pos[0]], [initial_pos[1]], [0], [0], [0], [0]], dtype=float)
        
        # state transition matrix: predicts next state from current state.
        self.F = np.array([[1, 0, 1, 0, 0.5, 0],
                          [0, 1, 0, 1, 0, 0.5],
                          [0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)
        
        # measurement matrix: maps state to measurement space.
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]], dtype=float)
        
        # process covariance: uncertainty in the model's prediction.
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 1.0, 1.0])
        
         # measurement covariance: uncertainty in the measurements.
        self.R = np.eye(2) * 2.0
        
        # error covariance: uncertainty in the state estimate.
        self.P = np.eye(6) * 100
     
    def predict(self):
        """
        the prediction step projects the state and its uncertainty forward in time.
        this is based on the system's dynamic model, before any new measurements are considered.
        """
        # project the a priori state estimate (x_k|k-1) using the state transition model.
        # x_k|k-1 = F * x_k-1|k-1

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (int(self.x[0, 0]), int(self.x[1, 0]))
    
    def update(self, measurement):
        """
        The update method is where the Kalman filter corrects its guess
        by using a new measurement. It smartly decides how much to trust
        the new data versus its own prediction, helping to get a more
        accurate and stable result.
        """
        # measurement vector (z_k): the new observation at time k.
        # innovation (y): the difference between the actual and predicted measurement. y = z - H * x_k|k-1
        # innovation covariance (S): the uncertainty of the innovation. S = H * P_k|k-1 * H^T + R
        # kalman gain (K): the optimal weighting factor to combine the prediction and measurement. K = P_k|k-1 * H^T * inv(S)
        # update the a posteriori state estimate (x_k|k). x_k|k = x_k|k-1 + K * y
        # update the a posteriori error covariance (P_k|k). P_k|k = (I - K * H) * P_k|k-1
        
        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
    def get_velocity(self):
        return (self.x[2, 0], self.x[3, 0])
    
    def get_acceleration(self):
        return (self.x[4, 0], self.x[5, 0])