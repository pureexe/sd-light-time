import numpy as np

def vector_to_angles(vectors):
    """
    Converts unit vectors to spherical coordinates (theta, phi).

    Parameters:
    vectors (numpy.ndarray): Input array of shape (..., 3), representing unit vectors.

    Returns:
    tuple: A tuple containing two arrays:
        - theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
        - phi (numpy.ndarray): Array of phi values in the range [0, 2*pi].
    """
    # Ensure input is a numpy array
    vectors = np.asarray(vectors)

    # Validate shape
    if vectors.shape[-1] != 3:
        raise ValueError("Input must have shape (..., 3).")

    # Extract components of the vectors
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]

    # Calculate theta (latitude angle)
    theta = np.arcsin(y)  # arcsin gives range [-pi/2, pi/2]

    # Calculate phi (longitude angle)
    phi = np.arctan2(-x, z)  # atan2 accounts for correct quadrant
    phi = (phi + 2 * np.pi) % (2 * np.pi)  # Normalize phi to range [0, 2*pi]

    return theta, phi

# Example usage:
unit_vectors = np.array([
    [0, 1, 0],  # Expected: theta = pi/2, phi = pi
    [0, 0, 1],  # Expected: theta = 0, phi = 0
    [1, 0, 0],  # Expected: theta = 0, phi = 3*pi/2
])

theta, phi = vector_to_angles(unit_vectors)
print("Theta:", theta)
print("Phi:", phi)
