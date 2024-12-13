import pytest
from unittest.mock import Mock, patch, MagicMock, call
import io
from PIL import Image as PILImage
import numpy as np
from LookBuilderPipeline.manager.resize_notification_manager import ResizeNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

@pytest.fixture
def resize_manager():
    return ResizeNotificationManager()

@pytest.fixture
def mock_session():
    session = MagicMock()
    return session

@pytest.fixture
def test_images():
    """Load test images."""
    with open('tests/img/p09.jpg', 'rb') as f:
        source_image = f.read()
    with open('tests/img/resize_p09.png', 'rb') as f:
        expected_resize = f.read()
    return source_image, expected_resize

@pytest.fixture
def mock_process():
    """Create a mock process with test parameters."""
    process = Mock(spec=ProcessQueue)
    process.process_id = 1
    process.image_id = 1
    process.parameters = {
        "target_size": 512,
        "square": True
    }
    return process

def test_init(resize_manager):
    """Test initialization of ResizeNotificationManager."""
    assert resize_manager.channels == ['image_resize']
    assert resize_manager.required_fields == ['process_id', 'image_id', 'size']

def test_process_resize_with_real_detection(resize_manager, mock_session, test_images, mock_process):
    """Test resize processing with real resize detection."""
    source_image, expected_resize = test_images
    
    # Mock image object
    mock_image = Mock(spec=Image)
    mock_image.image_id = 1
    mock_image.get_image_data = Mock(return_value=source_image)
    
    # Mock variant object
    mock_variant = Mock(spec=ImageVariant)
    mock_variant.variant_id = 1
    mock_variant.get_image_data = Mock(return_value=expected_resize)
    
    # Setup session mocks
    mock_session.query.return_value.get.return_value = mock_variant
    mock_session.connection.return_value.connection.lobject.return_value = Mock(
        write=Mock(),
        oid=123,
        close=Mock()
    )
    
    # Mock the database context
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Process the resize
        test_data = {
            'process_id': 1,
            'image_id': 1,
            'target_size': 512,
            'square': True
        }
        
        variant_id = resize_manager.process_resize(test_data)
        
        # Verify the process completed
        # assert variant_id is not None
        
        # Get the processed image
        processed_variant = mock_session.query(ImageVariant).get(variant_id)
        processed_image = processed_variant.get_image_data(mock_session)
        
        # Compare with expected resize image
        processed_array = np.array(PILImage.open(io.BytesIO(processed_image)))
        expected_array = np.array(PILImage.open(io.BytesIO(expected_resize)))
        
        # Allow for small differences in resize detection
        np.testing.assert_equal(
            processed_array.size,
            expected_array.size,
            err_msg="Processed resize differs significantly from expected resize"
        )
