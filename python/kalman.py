import numpy as np
from scipy.linalg import solve, LinAlgError

def create_state_vector(bbox):
    x1, y1, x2, y2 = bbox
    # zero velocity
    dx1, dy1, dx2, dy2 = 0.0, 0.0, 0.0, 0.0
    state_vector = np.array([x1, y1, x2, y2, dx1, dy1, dx2, dy2], dtype=float)
    return state_vector
