import numpy as np

def generate_sphere_map(image_size):
    """
    Generate a tuple of numpy arrays representing the latitude and longitude
    of a spherical map in a circular shape.

    Parameters:
        image_size (int): The size (width and height) of the square image.

    Returns:
        tuple: A tuple containing two numpy arrays (latitude, longitude).
    """
    # Create a grid of x, y coordinates
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xv, yv = np.meshgrid(x, y)

    # Compute the radius from the center
    radius = np.sqrt(xv**2 + yv**2)

    # Mask for locations outside the sphere
    mask = radius <= 1

    # Normalize coordinates within the circle to sphere coordinates
    theta = np.arctan2(yv, xv)  # Longitude: [0, 2*pi]
    phi = np.arccos(np.clip(radius, 0, 1))  # Latitude: [0, pi]

    # Adjust ranges: longitude [0, 2*pi], latitude [pi/2, -pi/2]
    longitude = (theta + 2 * np.pi) % (2 * np.pi)
    latitude = np.pi / 2 - phi

    # Set values outside the sphere to 0
    latitude[~mask] = 0
    longitude[~mask] = 0

    return latitude, longitude

# Example usage
if __name__ == "__main__":
    size = 512
    lat, lon = generate_sphere_map(size)
    print("Latitude shape:", lat.shape)
    print("Longitude shape:", lon.shape)
