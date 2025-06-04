import unittest
import numpy as np
from multiply_albedo_diffusionlight import to_linear_rgb, render_with_albedo

class TestImageFunctions(unittest.TestCase):

    def test_to_linear_rgb(self):
        image = np.array([[0.0, 0.5, 1.0]])
        expected = np.power(image, 2.4)
        result = to_linear_rgb(image)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_to_linear_rgb_with_custom_gamma(self):
        image = np.array([[0.25, 0.75]])
        gamma = 2.0
        expected = np.power(image, gamma)
        result = to_linear_rgb(image, gamma)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_render_with_albedo(self):
        shading = np.array([[1.0, 0.5]])
        albedo = np.array([[np.pi, np.pi / 2]])
        expected = (albedo / np.pi) * shading
        result = render_with_albedo(shading, albedo)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_render_with_zeros(self):
        shading = np.zeros((2,))
        albedo = np.ones((2,))
        expected = np.zeros((2,))
        result = render_with_albedo(shading, albedo)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
    print("All tests passed successfully.")