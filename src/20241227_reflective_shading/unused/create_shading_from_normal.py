
import torch 
import numpy as np

def get_light_vector(normal_map, viewing_vector=None):
    """
    Calculate the light vector L using the normal map N and viewing vector V.

    Args:
        normal_map (torch.Tensor): Tensor of shape [3, H, W] representing the normal map.
        viewing_vector (torch.Tensor, optional): Tensor of shape [3] representing the viewing vector.
                                         Defaults to [0, 0, 1].

    Returns:
        torch.Tensor: Light vector tensor of shape [3, H, W].
    """
    if viewing_vector is None:
        viewing_vector = torch.tensor([0.0, 0.0, 1.0], dtype=normal_map.dtype, device=normal_map.device)

    # Ensure viewing_vector is a column vector for broadcasting
    viewing_vector = viewing_vector.view(3, 1, 1)

    # Normalize the normal map and viewing vector
    normal_map = normal_map / torch.norm(normal_map, dim=0, keepdim=True).clamp(min=1e-8)
    viewing_vector = viewing_vector / torch.norm(viewing_vector).clamp(min=1e-8)

    # Compute the reflection vector R = 2 * (N . V) * N - V
    dot_product = torch.sum(normal_map * viewing_vector, dim=0, keepdim=True)  # N . V
    reflection_vector = 2 * dot_product * normal_map - viewing_vector

    # Normalize the reflection vector to get the light vector
    light_vector = reflection_vector / torch.norm(reflection_vector, dim=0, keepdim=True).clamp(min=1e-8)

    return light_vector

# Test functions
def test_get_light_vector():
    """Test cases to verify the correctness of the get_light_vector function."""
    # Test 1: Simple case with N = [0, 0, 1] and default V
    normal_map = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    expected_light_vector = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    light_vector = get_light_vector(normal_map)
    assert torch.allclose(light_vector, expected_light_vector, atol=1e-6), "Test 1 failed"

    # Test 2: Normal map N = [1/sqrt(2), 0, 1/sqrt(2)] and default V
    normal_map = torch.tensor([[[1.0]], [[0.0]], [[1.0]]]) / torch.sqrt(torch.tensor(2.0))
    normal_map = normal_map.float()
    expected_light_vector = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])
    light_vector = get_light_vector(normal_map)
    assert torch.allclose(light_vector, expected_light_vector, atol=1e-6), "Test 2 failed"

    # Test 3: Custom viewing vector V = [0, 1, 0] and N = [0, 1, 0]
    normal_map = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    viewing_vector = torch.tensor([0.0, 1.0, 0.0])
    expected_light_vector = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    light_vector = get_light_vector(normal_map, viewing_vector)
    assert torch.allclose(light_vector, expected_light_vector, atol=1e-6), "Test 3 failed"

    # Test 4: N and V are orthogonal
    normal_map = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])
    viewing_vector = torch.tensor([0.0, 0.0, 1.0])
    light_vector = get_light_vector(normal_map, viewing_vector)
    assert torch.allclose(light_vector, -viewing_vector.view(3, 1, 1), atol=1e-6), "Test 4 failed"


def main():
    test_get_light_vector() 
    print("All tests passed!")

if __name__ == "__main__":
    main()