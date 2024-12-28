import torch
import torch.nn.functional as F

def lambertian_render(normal_map, albedo, env_map, light_dir):
    """
    Perform Lambertian rendering given a normal map, albedo, equirectangular environment map,
    and light direction.

    Args:
        normal_map (torch.Tensor): Normal map of shape [3, H, W].
        albedo (torch.Tensor): Albedo map of shape [3, H, W].
        env_map (torch.Tensor): Equirectangular environment map of shape [3, A, 2A].
        light_dir (torch.Tensor): Light direction map of shape [3, H, W].

    Returns:
        torch.Tensor: Rendered image of shape [3, H, W].
    """
    # Ensure the inputs are valid
    assert normal_map.shape[0] == 3, "Normal map must have 3 channels."
    assert albedo.shape[0] == 3, "Albedo must have 3 channels."
    assert env_map.shape[0] == 3, "Environment map must have 3 channels."
    assert light_dir.shape[0] == 3, "Light direction must have 3 channels."

    # Normalize the normal map and light direction to unit vectors
    normal_map = F.normalize(normal_map, dim=0)
    light_dir = F.normalize(light_dir, dim=0)

    # Compute dot product of normal_map and light_dir
    dot_products = torch.sum(normal_map * light_dir, dim=0).clamp(min=0)  # [H, W]

    # Convert light direction to spherical coordinates for environment map lookup
    x, y, z = light_dir
    theta = torch.acos(z.clamp(-1, 1))  # [H, W], latitude
    phi = torch.atan2(y, x)  # [H, W], longitude
    phi = (phi + 2 * torch.pi) % (2 * torch.pi)  # Ensure phi is in [0, 2*pi]

    # Map spherical coordinates to environment map pixel indices
    A, B = env_map.shape[1:]  # A and 2A dimensions
    u = (phi / (2 * torch.pi) * (2 * A)).long()  # Longitude to pixel index
    v = (theta / torch.pi * A).long()  # Latitude to pixel index

    # Clamp indices to valid range
    u = u.clamp(0, 2 * A - 1)
    v = v.clamp(0, A - 1)

    # Sample environment map for each pixel in light direction
    env_map_sampled = env_map[:, v, u]  # [3, H, W]

    # Compute irradiance from dot product and sampled environment map
    irradiance = dot_products.unsqueeze(0) * env_map_sampled  # [3, H, W]

    # Compute final rendered image
    rendered_image = irradiance * albedo

    return rendered_image

def test_lambertian_render():
    """
    Test the lambertian_render function with synthetic data.
    """
    H, W = 64, 128
    A = 64

    # Generate synthetic normal map (pointing up in Z)
    normal_map = torch.zeros(3, H, W)
    normal_map[2, :, :] = 1.0

    # Generate synthetic albedo (white color)
    albedo = torch.ones(3, H, W)

    # Generate synthetic environment map (uniform lighting)
    env_map = torch.ones(3, A, 2 * A)

    # Generate synthetic light direction (pointing down in Z)
    light_dir = torch.zeros(3, H, W)
    light_dir[2, :, :] = -1.0

    # Perform rendering
    rendered_image = lambertian_render(normal_map, albedo, env_map, light_dir)

    # Assertions to validate output
    assert rendered_image.shape == (3, H, W), "Rendered image has incorrect shape."
    assert torch.all(rendered_image >= 0), "Rendered image has negative values."
    assert torch.all(rendered_image <= 1), "Rendered image exceeds expected range."

    print("Test passed!")

# Run the test
test_lambertian_render()
