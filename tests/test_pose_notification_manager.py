from unittest.mock import MagicMock, patch
import unittest
from LookBuilderPipeline.manager.pose_notification_manager import PoseNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

class TestPoseNotificationManager(unittest.TestCase):
    
    def setUp(self):
        # Initialize PoseNotificationManager or any required setup
        self.manager = PoseNotificationManager()
        self.manager.get_managed_session = MagicMock()  # Mock the session method

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_handle_notification_success(self, mock_logging):
        # Mock the session and query behavior
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock the process queue return value
        mock_process = MagicMock()
        mock_process.parameters = {'face': 'test_face'}
        mock_session.query.return_value.get.return_value = mock_process
        
        # Call the method
        result = self.manager.handle_notification('image_pose', {'process_id': 1, 'image_id': 2})
        
        # Assertions
        self.assertIsNotNone(result)
        mock_logging.info.assert_called_with("Processing pose with parameters: {'process_id': 1, 'image_id': 2, 'face': 'test_face'}")

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_handle_notification_process_not_found(self, mock_logging):
        # Mock the session and query behavior
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock the process queue return value to be None
        mock_session.query.return_value.get.return_value = None
        
        # Call the method
        result = self.manager.handle_notification('image_pose', {'process_id': 1, 'image_id': 2})
        
        # Assertions
        self.assertIsNone(result)
        mock_logging.error.assert_called_with("Process 1 not found or has no parameters")

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_process_pose_success(self, mock_logging):
        # Mock the session and query behavior
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock the image return value
        mock_image = MagicMock()
        mock_image.image_id = 2
        mock_session.query.return_value.get.return_value = mock_image
        
        # Mock the ImageVariant behavior
        mock_variant = MagicMock()
        mock_variant.get_or_create_variant.return_value = mock_variant
        with patch('LookBuilderPipeline.models.image_variant.ImageVariant', return_value=mock_variant):
            result = self.manager.process_pose({'process_id': 1, 'image_id': 2, 'face': 'test_face'})
        
        # Assertions
        self.assertIsNotNone(result)
        mock_logging.info.assert_called_with("Processing pose with parameters: {'process_id': 1, 'image_id': 2, 'face': 'test_face'}")

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_mark_process_error(self, mock_logging):
        # Mock the session and query behavior
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock the process queue return value
        mock_process = MagicMock()
        mock_session.query.return_value.get.return_value = mock_process
        
        # Call the method
        self.manager.mark_process_error(mock_session, 1, "Test error message")
        
        # Assertions
        mock_process.status = 'error'
        mock_process.error_message = "Test error message"
        mock_session.commit.assert_called_once()


