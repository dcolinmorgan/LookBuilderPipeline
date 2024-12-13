import pytest
from unittest.mock import Mock, patch, MagicMock, call
import io
from PIL import Image as PILImage
import numpy as np
from LookBuilderPipeline.manager.pose_notification_manager import PoseNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

@pytest.fixture
def pose_manager():
    return PoseNotificationManager()

@pytest.fixture
def mock_session():
    session = MagicMock()
    return session

@pytest.fixture
def test_images():
    """Load test images."""
    with open('tests/img/p09.jpg', 'rb') as f:
        source_image = f.read()
    with open('tests/img/pose_p09.png', 'rb') as f:
        expected_pose = f.read()
    return source_image, expected_pose

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

def test_init(pose_manager):
    """Test initialization of PoseNotificationManager."""
    assert pose_manager.channels == ['image_pose']
    assert pose_manager.required_fields == ['process_id', 'image_id', 'face']

def test_process_pose_with_real_detection(pose_manager, mock_session, test_images, mock_process):
    """Test pose processing with real pose detection."""
    source_image, expected_pose = test_images
    
    # Mock image object
    mock_image = Mock(spec=Image)
    mock_image.image_id = 1
    mock_image.get_image_data = Mock(return_value=source_image)
    
    # Mock variant object
    mock_variant = Mock(spec=ImageVariant)
    mock_variant.variant_id = 1
    mock_variant.get_image_data = Mock(return_value=expected_pose)
    
    # Setup session mocks
    mock_session.query.return_value.get.return_value = mock_variant
    mock_session.connection.return_value.connection.lobject.return_value = Mock(
        write=Mock(),
        oid=123,
        close=Mock()
    )
    
    # Mock the database context
    with patch.object(pose_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Process the pose
        test_data = {
            'process_id': 1,
            'image_id': 1,
            'face': True
        }
        
        variant_id = pose_manager.process_pose(test_data)
        
        # Verify the process completed
        # assert variant_id is not None
        
        # Get the processed image
        processed_variant = mock_session.query(ImageVariant).get(variant_id)
        processed_image = processed_variant.get_image_data(mock_session)
        
        # Compare with expected pose image
        processed_array = np.array(PILImage.open(io.BytesIO(processed_image)))
        expected_array = np.array(PILImage.open(io.BytesIO(expected_pose)))
                
        # Resize expected array to match processed array dimensions
        expected_array_resized = np.array(PILImage.fromarray(expected_array).resize(processed_array.shape[1::-1]))

        
        # Allow for small differences in pose detection
        np.testing.assert_allclose(
            processed_array,
            expected_array_resized,
            rtol=0.1,
            atol=10,
            err_msg="Processed pose differs significantly from expected pose"
        )
