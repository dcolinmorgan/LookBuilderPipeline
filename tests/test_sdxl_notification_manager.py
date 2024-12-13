import pytest
from unittest.mock import Mock, patch, MagicMock, call
import io
from PIL import Image as PILImage
import numpy as np
from LookBuilderPipeline.manager.sdxl_notification_manager import SDXLNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

@pytest.fixture
def sdxl_manager():
    return SDXLNotificationManager()

@pytest.fixture
def mock_session():
    session = MagicMock()
    return session

@pytest.fixture
def test_images():
    """Load test images."""
    with open('tests/img/p09.jpg', 'rb') as f:
        source_image = f.read()
    with open('tests/img/segment_p09.png', 'rb') as f:
        expected_sdxl = f.read()
    return source_image, expected_sdxl

@pytest.fixture
def mock_process():
    """Create a mock process with test parameters."""
    process = Mock(spec=ProcessQueue)
    process.process_id = 1
    process.image_id = 1
    process.parameters = {
        "face": True
    }
    return process

def test_init(sdxl_manager):
    """Test initialization of SDXLNotificationManager."""
    assert sdxl_manager.channels == ['image_sdxl']
    assert sdxl_manager.required_fields == ['process_id', 'image_id', 'prompt']

def test_process_sdxl_with_real_detection(sdxl_manager, mock_session, test_images, mock_process):
    """Test sdxl processing with real sdxl detection."""
    source_image, expected_sdxl = test_images
    
    # Mock image object
    mock_image = Mock(spec=Image)
    mock_image.image_id = 1
    mock_image.get_image_data = Mock(return_value=source_image)
    
    # Mock variant object
    mock_variant = Mock(spec=ImageVariant)
    mock_variant.image_id = 1
    mock_variant.get_image_data = Mock(return_value=expected_sdxl)
    
    # Setup session mocks
    mock_session.query.return_value.get.return_value = mock_variant
    mock_session.connection.return_value.connection.lobject.return_value = Mock(
        write=Mock(),
        oid=123,
        close=Mock()
    )
    
    # Mock the database context
    with patch.object(sdxl_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Process the sdxl
        test_data = {
            'process_id': 1,
            'image_id': 1,
            'prompt': 'a beautiful woman on the beach',
            'negative_prompt': 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves',
            'seed': 123456,
            'steps': 20,
            'guidance_scale': 7,
            'strength': 0.95,
            'LoRA': '',
        }
        
        variant_id = sdxl_manager.process_sdxl(test_data)
        
        # Verify the process completed
        # assert variant_id is not None
        
        # Get the processed image
        processed_variant = mock_session.query(ImageVariant).get(variant_id)
        processed_image = processed_variant.get_image_data(mock_session)
        
        A=PILImage.open(io.BytesIO(processed_image))
        A.save('processed_sdxl.png')
        from LookBuilderPipeline.segment import segment_image
        _,A = segment_image(A, inverse=True)
        processed_array = np.array(A)
        
        # Compare with expected sdxl image
        # processed_array = np.array(PILImage.open(io.BytesIO(processed_image)))
        expected_array = np.array(PILImage.open(io.BytesIO(expected_sdxl)))
                
        # Resize expected array to match processed array dimensions
        expected_array_resized = np.array(PILImage.fromarray(expected_array).resize(processed_array.shape[1::-1]))

        # TODO: fix this test using QA
        # Allow for small differences in sdxl detection
        np.testing.assert_allclose(
            processed_array,
            expected_array_resized,
            rtol=0.1,
            atol=10,
            err_msg="Processed sdxl differs significantly from expected sdxl"
        )
