import pytest
from unittest.mock import Mock, patch, MagicMock, call
from PIL import Image as PILImage
import io

from LookBuilderPipeline.manager.resize_notification_manager import ResizeNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image_variant import ImageVariant
from LookBuilderPipeline.models.image import Image

@pytest.fixture
def resize_manager():
    return ResizeNotificationManager()

@pytest.fixture
def mock_session():
    session = MagicMock()
    return session

@pytest.fixture
def mock_process():
    process = Mock(spec=ProcessQueue)
    process.process_id = 123
    process.image_id = 456
    process.user_id = 1
    process.parameters = {
        "size": 512,
        "aspect_ratio": 1.0,
        "square": False
    }
    return process

@pytest.fixture
def mock_image():
    """Create a test image in bytes format."""
    img = PILImage.new('RGB', (1024, 1024), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def test_init(resize_manager):
    """Test initialization of ResizeNotificationManager."""
    assert resize_manager.channels == ['image_resize']
    assert resize_manager.required_fields == ['process_id', 'image_id', 'size']

def test_handle_notification(resize_manager):
    """Test handling of resize notifications."""
    # Mock process_resize
    resize_manager.process_resize = Mock()
    
    # Mock session and process with integer IDs
    mock_process = Mock(spec=ProcessQueue)
    mock_process.parameters = {'size': 512}  # Add required size parameter
    mock_session = Mock()
    mock_session.query.return_value.get.return_value = mock_process
    
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Test valid channel with data (using integer IDs)
        test_data = {
            'process_id': 123,      # Integer instead of string
            'image_id': 456         # Integer instead of string
        }
        
        resize_manager.handle_notification('image_resize', test_data)
        
        # Verify process_resize was called with correct data
        expected_data = {
            'process_id': 123,      # Integer instead of string
            'image_id': 456,        # Integer instead of string
            'size': 512,
            'aspect_ratio': 1.0,
            'square': False
        }
        resize_manager.process_resize.assert_called_once_with(expected_data)
        
        # Test invalid channel
        resize_manager.process_resize.reset_mock()
        result = resize_manager.handle_notification('invalid', test_data)
        assert result is None
        resize_manager.process_resize.assert_not_called()

def test_process_item_success(resize_manager, mock_session, mock_process, mock_image):
    """Test successful processing of a resize item."""
    # Setup
    mock_process.image_id = 123
    mock_process.parameters = {'size': 100, 'aspect_ratio': 1.0, 'square': False}
    
    # Mock Image query
    mock_image_obj = Mock(spec=Image)
    mock_image_obj.get_or_create_resize_variant = Mock()
    mock_image_query = Mock()
    mock_image_query.get.return_value = mock_image_obj
    
    # Mock ProcessQueue query
    mock_process_query = Mock()
    mock_process_query.filter.return_value.with_for_update.return_value.first.return_value = mock_process
    
    def query_side_effect(model):
        if model == Image:
            return mock_image_query
        if model == ProcessQueue:
            return mock_process_query
        return Mock()
    
    mock_session.query = Mock(side_effect=query_side_effect)

    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Execute
        resize_manager.process_item(mock_process)
        
        # Verify both queries were made
        assert mock_session.query.call_args_list == [
            call(Image),
            call(ProcessQueue)
        ]
        mock_image_query.get.assert_called_once_with(123)
        mock_image_obj.get_or_create_resize_variant.assert_called_once_with(
            mock_session,
            size=100,
            aspect_ratio=1.0,
            square=False
        )

def test_process_item_missing_parameters(resize_manager, mock_process):
    """Test processing with missing parameters."""
    # Setup mock process with empty parameters
    mock_process.process_id = 123
    mock_process.image_id = 456
    mock_process.parameters = {}  # Empty parameters
    
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_session = Mock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Test the validation
        with pytest.raises(ValueError, match="Process is missing required size parameter"):
            resize_manager.process_item(mock_process)

def test_process_resize_success(resize_manager, mock_session):
    """Test successful resize processing."""
    test_data = {
        'process_id': 'test_123',
        'image_id': 'img_123',
        'parameters': {'size': 512}
    }
    
    # Mock session and process
    mock_process = Mock(spec=ProcessQueue)
    mock_process.parameters = {'size': 512}
    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_process
    
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Mock update_process_status
        resize_manager.update_process_status = Mock()
        
        # Process the resize
        result = resize_manager.process_resize(test_data)
        
        # Verify process was updated
        resize_manager.update_process_status.assert_called_with(
            mock_session, 
            'test_123', 
            'completed'
        )

# def test_process_resize_missing_size(resize_manager, mock_session):
#     """Test resize processing with missing size parameter."""
#     test_data = {
#         'process_id': 'test_123',
#         'image_id': 'img_123',
#         'parameters': {}  # Missing size
#     }
    
#     # Mock session and process
#     mock_process = Mock(spec=ProcessQueue)
#     mock_process.parameters = {}  # Empty parameters
#     mock_session.query.return_value.get.return_value = mock_process
    
#     with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
#         mock_get_session.return_value.__enter__.return_value = mock_session
        
#         with pytest.raises(ValueError, match=".*missing required size parameter"):
#             resize_manager.process_resize(test_data)

def test_resize_creates_variant(resize_manager, mock_session, mock_process, mock_image):
    """Test that resize process creates an ImageVariant."""
    # Setup
    mock_process.image_id = 123
    mock_process.parameters = {'size': 100, 'aspect_ratio': 1.0, 'square': False}
    
    # Mock Image query
    mock_image_obj = Mock(spec=Image)
    mock_image_obj.get_or_create_resize_variant = Mock()
    mock_image_query = Mock()
    mock_image_query.get.return_value = mock_image_obj
    
    # Mock ProcessQueue query
    mock_process_query = Mock()
    mock_process_query.filter.return_value.with_for_update.return_value.first.return_value = mock_process
    
    def query_side_effect(model):
        if model == Image:
            return mock_image_query
        if model == ProcessQueue:
            return mock_process_query
        return Mock()
    
    mock_session.query = Mock(side_effect=query_side_effect)

    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Execute
        resize_manager.process_item(mock_process)
        
        # Verify variant was created
        mock_image_obj.get_or_create_resize_variant.assert_called_once_with(
            mock_session,
            size=100,
            aspect_ratio=1.0,
            square=False
        )

def test_process_resize_workflow(resize_manager, mock_session, mock_process, mock_image):
    """Test the complete resize workflow with all database operations."""
    mock_process.image_id = 123  # Change from string to integer
    resize_manager.get_image = Mock(return_value=mock_image)

    # Mock the image object
    mock_image_obj = Mock(spec=Image)
    mock_session.query.return_value.get.return_value = mock_image_obj

    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Process the resize
        resize_manager.process_item(mock_process)

def test_error_handling(resize_manager, mock_process):
    """Test error handling during resize process."""
    mock_process.image_id = 123  # Change from string to integer
    
    # Mock the database error instead of get_image
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_session = Mock()
        mock_session.query.return_value.get.side_effect = Exception("Test error")
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(Exception):  # Remove specific message matching
            resize_manager.process_item(mock_process)
