import pytest
from diffusers.utils import load_image
from LookBuilderPipeline.LookBuilderPipeline.segment import segment_image
from PIL import Image
import numpy as np

def test_segment_image_returns_tuple():
    result = segment_image("LookBuilderPipeline/img/p09.jpg")
    assert isinstance(result, tuple)
    assert len(result) == 3

def test_segment_image_output_size():
    img=load_image("LookBuilderPipeline/img/p09.jpg")
    segmented_outfit, mask, mask_array = segment_image("LookBuilderPipeline/img/p09.jpg")
    assert segmented_outfit.size == img.size
    assert mask_array.T.shape == img.size
    assert mask_array.dtype == np.uint8

def test_segment_image_invalid_path():
    with pytest.raises(Exception):
        segment_image("non_existent_image.jpg")

@pytest.mark.parametrize("additional_option", [None, "shoe", "bag"])
def test_segment_image_various_options(additional_option):
    segmented_outfit, mask, mask_array = segment_image("LookBuilderPipeline/img/p09.jpg", additional_option=additional_option)
    assert isinstance(segmented_outfit, Image.Image)
    assert isinstance(mask_array, np.ndarray)

def test_segment_image_consistency():
    result1 = segment_image("LookBuilderPipeline/img/p09.jpg")
    result2 = segment_image("LookBuilderPipeline/img/p09.jpg")
    np.testing.assert_array_equal(result1[1], result2[1])

@pytest.mark.parametrize("size", [(512), (1024), (824)])
def test_segment_image_different_sizes(size):
    segmented_outfit, mask, mask_array = segment_image("LookBuilderPipeline/img/p09.jpg", resize=True,size=size)
    assert segmented_outfit.size[0] == size
