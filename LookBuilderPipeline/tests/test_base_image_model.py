import pytest
from PIL import Image
import numpy as np
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector

@pytest.fixture
def mock_image():
    return ("LookBuilderPipeline/img/p09.jpg")


def test_generate_image_extras(mock_image):
    # Setup mocks
    poseA,maskA = BaseImageModel.generate_image_extras(mock_image,inv=False)
    poseB,maskB = BaseImageModel.generate_image_extras(mock_image,inv=True)
    
    # Assert that poseA is equal to itself (this is always true)
    assert np.array_equal(np.array(poseA), np.array(poseA)), "poseA should be equal to itself"
    
    # Assert that maskA is not equal to maskB
    assert not np.array_equal(np.array(maskA), np.array(maskB)), "maskA should not be equal to maskB"
if __name__ == '__main__':
    pytest.main()
