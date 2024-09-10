import pytest
from LookBuilderPipeline.segmentation.segmentation import segment_image
from PIL import Image
import numpy as np

def test_segment_image_returns_tuple():
    # Test that the function returns a tuple
    result = segment_image("img/p09.jpg")
    assert isinstance(result, tuple)
    assert len(result) == 2  # Should return (segmented_outfit, mask)

def test_segment_image_output_size():
    # Test that the output image size matches the input image size
    input_image = Image.new('RGB', (512, 512))
    input_image.save("test_input.jpg")
    segmented_outfit, mask = segment_image("test_input.jpg")
    assert segmented_outfit.size == (512, 512)

def test_segment_image_invalid_path():
    # Test that the function raises an exception for an invalid file path
    with pytest.raises(Exception):
        segment_image("non_existent_image.jpg")

@pytest.mark.parametrize("additional_option", [None, "shoe", "bag"])
def test_segment_image_various_options(additional_option):
    # Test that the function works with various additional options
    input_image = Image.new('RGB', (512, 512))
    input_image.save("test_input.jpg")
    segmented_outfit, mask = segment_image("test_input.jpg", additional_option=additional_option)
    assert isinstance(segmented_outfit, Image.Image)
    assert isinstance(mask, np.ndarray)

def test_segment_image_mask_output():
    input_image = Image.new('RGB', (100, 100))
    input_image.save("test_input.jpg")
    _, mask = segment_image("test_input.jpg")
    assert mask.shape == (100, 100)
    assert mask.dtype == np.bool_ 

def test_segment_image_consistency():
    input_image = Image.new('RGB', (100, 100))
    input_image.save("test_input.jpg")
    result1 = segment_image("test_input.jpg")
    result2 = segment_image("test_input.jpg")
    np.testing.assert_array_equal(result1[1], result2[1])

@pytest.mark.parametrize("size", [(100, 100), (200, 300), (500, 500)])
def test_segment_image_different_sizes(size):
    input_image = Image.new('RGB', size)
    input_image.save("test_input.jpg")
    segmented_outfit, mask = segment_image("test_input.jpg", resize=size)
    assert segmented_outfit.size == size
