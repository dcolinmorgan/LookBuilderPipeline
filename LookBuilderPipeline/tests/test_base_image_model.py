import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector

@pytest.fixture
def mock_image():
    return Image.new('RGB', (100, 100))

@patch('OpenposeDetector')
@patch('CannyDetector')
def test_generate_image_extras(mock_show_images, mock_canny, mock_pipeline, mock_openpose, mock_image):
    # Setup mocks
    mock_openpose.from_pretrained.return_value.return_value = mock_image
    mock_pipeline.return_value.return_value = [{'mask': np.zeros((100, 100)), 'label': 'Upper-clothes'}]
    mock_canny.return_value.return_value = mock_image
    
    # Call the method
    final_mask, canny_image, pose_image = BaseImageModel.generate_image_extras(mock_image)
    
    # Assertions
    assert isinstance(final_mask, Image.Image)
    assert isinstance(canny_image, Image.Image)
    assert isinstance(pose_image, Image.Image)
    
    # Check if mocks were called
    mock_openpose.from_pretrained.assert_called_once_with('lllyasviel/ControlNet')
    mock_pipeline.assert_called_once_with(model="mattmdjaga/segformer_b2_clothes")
    mock_canny.assert_called_once()
    mock_show_images.assert_called_once()

def test_generate_image_extras_segmentation():
    # Test segmentation logic
    with patch('LookBuilderPipeline.image_models.base_image_model.pipeline') as mock_pipeline:
        mock_pipeline.return_value.return_value = [
            {'mask': np.ones((100, 100)), 'label': 'Upper-clothes'},
            {'mask': np.ones((100, 100)), 'label': 'Pants'},
            {'mask': np.ones((100, 100)), 'label': 'Background'}
        ]
        
        mock_image = MagicMock()
        final_mask, _, _ = BaseImageModel.generate_image_extras(mock_image)
        
        # The final mask should be the sum of 'Upper-clothes' and 'Pants'
        assert np.array(final_mask).sum() == 200 * 100  # 2 layers of 100x100 ones

def test_generate_image_extras_random_label():
    # Test if a random label is generated
    with patch('numpy.random.randint') as mock_randint:
        mock_randint.return_value = 12345
        
        with patch('LookBuilderPipeline.image_models.base_image_model.showImagesHorizontally') as mock_show:
            BaseImageModel.generate_image_extras(MagicMock())
            
            mock_show.assert_called_once()
            args, _ = mock_show.call_args
            assert '12345' in args[1]  # Check if the random label is used in the filename

if __name__ == '__main__':
    pytest.main()
