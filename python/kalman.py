import numpy as np

class PointKalmanTracker:
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
    
class BBoxGroupKalmanTracker:
    def __init__(self, initial_bbox):
        """
        Initializes four Kalman trackers, one for each corner of the bounding box.
        initial_bbox: (x1, y1, x2, y2) -> top-left, bottom-right
        """
        x1, y1, x2, y2 = initial_bbox
        self.trackers = {
            "tl": PointKalmanTracker((x1, y1)),  # top-left
            "tr": PointKalmanTracker((x2, y1)),  # top-right
            "bl": PointKalmanTracker((x1, y2)),  # bottom-left
            "br": PointKalmanTracker((x2, y2)),  # bottom-right
        }

    def predict(self):
        """
        Predict the next positions of all four corners.
        Returns bbox in (x1, y1, x2, y2) format.
        """
        tl = self.trackers["tl"].predict()
        tr = self.trackers["tr"].predict()
        bl = self.trackers["bl"].predict()
        br = self.trackers["br"].predict()

        # reconstruct bbox: top-left and bottom-right
        x1 = int(min(tl[0], bl[0]))
        y1 = int(min(tl[1], tr[1]))
        x2 = int(max(tr[0], br[0]))
        y2 = int(max(bl[1], br[1]))

        return (x1, y1, x2, y2)

    def update(self, new_bbox):
        """
        Update all four corner trackers with the new bbox measurement.
        new_bbox: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = new_bbox
        self.trackers["tl"].update((x1, y1))
        self.trackers["tr"].update((x2, y1))
        self.trackers["bl"].update((x1, y2))
        self.trackers["br"].update((x2, y2))

    def get_bbox(self):
        """
        Return the current best estimate of the bbox
        based on the four tracked corners.
        """
        tl = self.trackers["tl"].x
        tr = self.trackers["tr"].x
        bl = self.trackers["bl"].x
        br = self.trackers["br"].x

        x1 = int(min(tl[0, 0], bl[0, 0]))
        y1 = int(min(tl[1, 0], tr[1, 0]))
        x2 = int(max(tr[0, 0], br[0, 0]))
        y2 = int(max(bl[1, 0], br[1, 0]))

        return (x1, y1, x2, y2)
    
    def get_velocities(self):
        """Return velocities for all four corners."""
        return {k: t.get_velocity() for k, t in self.trackers.items()}
    
    def get_accelerations(self):
        """Return accelerations for all four corners."""
        return {k: t.get_acceleration() for k, t in self.trackers.items()}
