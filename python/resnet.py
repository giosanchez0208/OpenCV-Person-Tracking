import numpy as np

def get_resnet_embedding(frame, bbox):
    """
    Return a dummy embedding (just random numbers for now)
    to simulate ResNet features.
    """
    # Example: 512-dim embedding
    embedding_dim = 512
    embedding = np.random.rand(embedding_dim).astype(float)
    return embedding
