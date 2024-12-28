import torch
import torch.nn.functional as F

def lambertian_render(normal_map, albedo, env_map):
    """
    Perform Lambertian rendering given a normal map, albedo, and equirectangular environment map.

    Args:
        normal_map (torch.Tensor): Normal map of shape [3, H, W].
        albedo (torch.Tensor): Albedo map of shape [3, H, W].
        env_map (torch.Tensor): Equirectangular environment map of shape [3, A, 2A].

    Returns:
        torch.Tensor: Rendered image of shape [3, H, W].
    """
    # Ensure the inputs are valid
    assert normal_map.shape[0] == 3, "Normal map must have 3 channels."
    assert albedo.shape[0] == 3, "Albedo must have 3 channels."
    assert env_map.shape[0] == 3, "Environment map must have 3 channels."

    # Normalize the normal map to unit vectors
    normal_map = F.normalize(normal_map, dim=0)

    # Convert equirectangular environment map to spherical directions
    A, B = env_map.shape[1:]  # A and 2A dimensions
    theta = torch.linspace(0, torch.pi, A, device=env_map.device)  # latitude
    phi = torch.linspace(0, 2 * torch.pi, 2 * A, device=env_map.device)  # longitude
    
    phi, theta = torch.meshgrid(phi, theta, indexing="xy")

    # Directional vectors from spherical coordinates
    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=0)  # Shape: [3, A, 2A]

    # Compute dot product of normal_map and light directions
    H, W = normal_map.shape[1:]
    normal_map_flat = normal_map.view(3, -1).t()  # [H*W, 3]
    directions_flat = directions.view(3, -1).t()  # [A*2A, 3]

    # Dot products for Lambertian shading
    dot_products = torch.mm(normal_map_flat, directions_flat.t()).clamp(min=0)  # [H*W, A*2A]

    # Integrate light contribution from environment map
    light_intensity = env_map.view(3, -1)  # [3, A*2A]
    irradiance = torch.matmul(dot_products, light_intensity.t())  # [H*W, 3]
    irradiance = irradiance.view(H, W, 3).permute(2, 0, 1)  # [3, H, W]

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

    # Perform rendering
    rendered_image = lambertian_render(normal_map, albedo, env_map)

    # Assertions to validate output
    assert rendered_image.shape == (3, H, W), "Rendered image has incorrect shape."
    assert torch.all(rendered_image >= 0), "Rendered image has negative values."
    assert torch.all(rendered_image <= 1), "Rendered image exceeds expected range."

    print("Test passed!")

# Run the test
test_lambertian_render()
