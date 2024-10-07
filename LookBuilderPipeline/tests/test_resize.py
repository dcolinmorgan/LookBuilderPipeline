import pytest
from diffusers.utils import load_image
from LookBuilderPipeline.resize import resize_images
from PIL import Image
import numpy as np

@pytest.mark.parametrize("size", [(512)])
def test_segment_image_returns_tuple(size):
    result = resize_images("LookBuilderPipeline/img/p09.jpg",target_size=size,aspect_ratio=None,square=False)
    assert isinstance(result, Image.Image)
    
@pytest.mark.parametrize("size", [(512), (1024), (824)])
def test_resize_image_different_sizes_square(size):
    new_image = resize_images("LookBuilderPipeline/img/p09.jpg",target_size=size,aspect_ratio=1/1,square=True)
    assert np.round(new_image.size[0],2) == np.round(new_image.size[1],2)

@pytest.mark.parametrize("size", [(512), (1024), (824)])
def test_segment_image_different_sizes_ar(size):
    new_image = resize_images("LookBuilderPipeline/img/p09.jpg",target_size=size,aspect_ratio=6/9,square=False)
    assert np.round(new_image.size[0] / new_image.size[1],2) == np.round(6/9,2)
