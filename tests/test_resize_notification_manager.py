import pytest
from unittest.mock import Mock, patch, MagicMock
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
    process.process_id = "test_process_123"
    process.image_id = "test_image_123"
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
    assert resize_manager.channels == ['resize']
    assert resize_manager.required_fields == ['process_id', 'image_id']

def test_handle_notification(resize_manager):
    """Test handling of resize notifications."""
    resize_manager.process_resize = Mock()
    test_data = {
        'process_id': 'test_123',
        'image_id': 'img_123',
        'parameters': {'size': 512}
    }
    
    # Test valid channel
    resize_manager.handle_notification('resize', test_data)
    resize_manager.process_resize.assert_called_once_with(test_data)
    
    # Test invalid channel
    result = resize_manager.handle_notification('invalid', test_data)
    assert result is None

def test_process_item_success(resize_manager, mock_session, mock_process, mock_image):
    """Test successful processing of a resize item."""
    # Mock the image
    mock_image_obj = Mock(spec=Image)
    mock_image_obj.get_or_create_resize_variant = Mock()
    mock_session.query.return_value.get.return_value = mock_image_obj

    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session

        # Process the item
        resize_manager.process_item(mock_process)

        # Verify image was retrieved
        mock_session.query.assert_called_with(Image)
        mock_session.query.return_value.get.assert_called_with(mock_process.image_id)

        # Verify resize variant was requested
        mock_image_obj.get_or_create_resize_variant.assert_called_once_with(
            mock_session,
            size=mock_process.parameters['size'],
            aspect_ratio=mock_process.parameters.get('aspect_ratio', 1.0),
            square=mock_process.parameters.get('square', False)
        )

        # Verify process was updated
        mock_session.commit.assert_called()

def test_process_item_missing_parameters(resize_manager, mock_process):
    """Test processing with missing parameters."""
    mock_process.parameters = {}  # Empty parameters
    
    with pytest.raises(ValueError, match=".*missing required size parameter"):
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

def test_process_resize_missing_size(resize_manager, mock_session):
    """Test resize processing with missing size parameter."""
    test_data = {
        'process_id': 'test_123',
        'image_id': 'img_123',
        'parameters': {}  # Missing size
    }
    
    # Mock session and process
    mock_process = Mock(spec=ProcessQueue)
    mock_process.parameters = {}
    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_process
    
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        with pytest.raises(ValueError, match=".*missing required size parameter"):
            resize_manager.process_resize(test_data)

def test_resize_creates_variant(resize_manager, mock_session, mock_process, mock_image):
    """Test that resize process creates an ImageVariant."""
    # Mock the get_image method
    resize_manager.get_image = Mock(return_value=mock_image)
    resize_manager.store_large_object = Mock(return_value=12345)  # Mock OID return
    
    # Mock the database query to return our mock process
    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_process
    mock_session.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = mock_process
    
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Process the resize
        resize_manager.process_item(mock_process)
        
        # Verify the image was retrieved and stored
        resize_manager.get_image.assert_called_once_with(mock_process.image_id)
        resize_manager.store_large_object.assert_called_once()
        
        # Verify the variant was created and stored
        add_calls = [
            call for call in mock_session.method_calls 
            if call[0] == 'add' and isinstance(call[1][0], ImageVariant)
        ]
        assert len(add_calls) >= 1, "No ImageVariant was added to the session"
        
        # Verify the commit happened
        mock_session.commit.assert_called()

def test_process_resize_workflow(resize_manager, mock_session, mock_process, mock_image):
    """Test the complete resize workflow with all database operations."""
    resize_manager.get_image = Mock(return_value=mock_image)
    resize_manager.store_large_object = Mock(return_value=12345)  # Mock OID return
    
    mock_session.query.return_value.filter_by.return_value.first.return_value = mock_process
    
    with patch.object(resize_manager, 'get_managed_session') as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        # Process the resize
        resize_manager.process_item(mock_process)
        
        # Get all database operations in order
        db_operations = mock_session.mock_calls
        
        # Verify key operations occurred
        operations_occurred = {
            'query': False,
            'add': False,
            'commit': False
        }
        
        for call in db_operations:
            method_name = call[0]
            if method_name == 'query':
                operations_occurred['query'] = True
            elif method_name == 'add' and isinstance(call[1][0], ImageVariant):
                operations_occurred['add'] = True
            elif method_name == 'commit':
                operations_occurred['commit'] = True
        
        # Verify all required operations occurred
        assert operations_occurred['query'], "No database query performed"
        assert operations_occurred['add'], "No ImageVariant added"
        assert operations_occurred['commit'], "No commit performed"

def test_error_handling(resize_manager, mock_process):
    """Test error handling during resize process."""
    resize_manager.get_image = Mock(side_effect=Exception("Test error"))
    
    with pytest.raises(Exception, match="Test error"):
        resize_manager.process_item(mock_process)