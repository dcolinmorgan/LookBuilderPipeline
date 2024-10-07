import pytest
from PIL import Image
import os
from LookBuilderPipeline.padding import pad_image

@pytest.fixture
def test_image_path():
    return "LookBuilderPipeline/img/p09.jpg"

def test_pad_image_returns_image(test_image_path):
    result = pad_image(test_image_path)
    assert isinstance(result, Image.Image)

def test_pad_image_output_size(test_image_path):
    result = pad_image(test_image_path, resize=True, size=(512, 512))
    assert result.size == (512, 512)

def test_pad_square_image(test_image_path):
    # Create a square test image
    test_image = Image.new("RGB", (100, 100), color="red")
    test_image_path = "test_square.jpg"
    test_image.save(test_image_path)
    
    result = pad_image(test_image_path)
    assert result.size == (100, 100)  # Should remain the same size
    
    os.remove(test_image_path)

def test_pad_rectangular_image(test_image_path):
    # Create a rectangular test image
    test_image = Image.new("RGB", (80, 120), color="blue")
    test_image_path = "test_rectangle.jpg"
    test_image.save(test_image_path)
    
    result = pad_image(test_image_path)
    assert result.size == (120, 120)  # Should be padded to square
    
    os.remove(test_image_path)

def test_pad_and_resize(test_image_path):
    # Create a rectangular test image
    test_image = Image.new("RGB", (80, 120), color="green")
    test_image_path = "test_resize.jpg"
    test_image.save(test_image_path)
    
    result = pad_image(test_image_path, resize=True, size=(256, 256))
    assert result.size == (256, 256)  # Should be padded and resized
    
    os.remove(test_image_path)

def test_pad_image_content(test_image_path):
    # Create a small test image with known content
    test_image = Image.new("RGB", (50, 30), color="purple")
    test_image_path = "test_content.jpg"
    test_image.save(test_image_path)
    
    result = pad_image(test_image_path)
    
    # Check if the original image is centered in the padded image
    assert result.size == (50, 50)
    assert result.getpixel((0, 0)) == (255, 255, 255)  # White padding in the corner
    
    os.remove(test_image_path)

def test_pad_image_different_sizes(test_image_path):
    sizes_to_test = [(100, 100), (512, 512), (1024, 1024)]
    test_image = Image.new("RGB", (60, 40), color="yellow")
    test_image_path = "test_sizes.jpg"
    test_image.save(test_image_path)
    
    for size in sizes_to_test:
        result = pad_image(test_image_path, resize=True, size=size)
        assert result.size == size
    
    os.remove(test_image_path)
