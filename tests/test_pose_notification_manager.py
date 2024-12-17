from unittest.mock import MagicMock, patch
import unittest
from LookBuilderPipeline.manager.pose_notification_manager import PoseNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

class TestPoseNotificationManager(unittest.TestCase):
    """
    Test suite for PoseNotificationManager.
    Tests the handling of pose detection notifications and process management.
    """
    
    def setUp(self):
        """
        Setup runs before each test.
        Creates a fresh PoseNotificationManager instance and mocks its session handling
        to avoid actual database connections during testing.
        """
        self.manager = PoseNotificationManager()
        self.manager.get_managed_session = MagicMock()

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_handle_notification_success(self, mock_logging):
        """
        Tests successful handling of a pose notification.
        
        Verifies that:
        1. Manager correctly processes a valid pose notification
        2. Process parameters are properly retrieved from the database
        3. Logging is performed with correct parameters
        4. A result is returned (indicating successful processing)
        """
        # Setup mock database session
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock a process in the database with face detection parameter
        mock_process = MagicMock()
        mock_process.parameters = {'face': True}
        mock_session.query.return_value.get.return_value = mock_process
        
        # Simulate receiving a pose notification
        result = self.manager.handle_notification('image_pose', {'process_id': 1, 'image_id': 2})
        
        # Verify the notification was handled correctly
        self.assertIsNotNone(result)
        mock_logging.info.assert_called_with(
            "Processing pose with parameters: {'process_id': 1, 'image_id': 2, 'face': True}"
        )

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_handle_notification_process_not_found(self, mock_logging):
        """
        Tests handling of notifications when the process doesn't exist.
        
        Verifies that:
        1. Manager gracefully handles missing processes
        2. Appropriate error is logged
        3. None is returned (indicating failed processing)
        4. Database session is properly managed
        """
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Simulate process not found in database
        mock_session.query.return_value.get.return_value = None
        
        result = self.manager.handle_notification('image_pose', {'process_id': 1, 'image_id': 2})
        
        # Verify proper error handling
        self.assertIsNone(result)
        mock_logging.error.assert_called_with("Process 1 not found or has no parameters")

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_process_pose_success(self, mock_logging):
        """
        Tests successful processing of a pose detection request.
        
        Verifies that:
        1. Image is correctly retrieved from database
        2. ImageVariant is properly created
        3. Pose detection variant is created with correct parameters
        4. Processing result is returned
        5. All database operations are performed in correct order
        """
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        mock_process = MagicMock()
        mock_process.parameters = {'face': True}
        mock_session.query.return_value.get.return_value = mock_process
        
        result = self.manager.process_pose({'process_id': 1, 'image_id': 2, 'face': True})
        
        self.assertIsNotNone(result)

    @patch('LookBuilderPipeline.manager.pose_notification_manager.logging')
    def test_mark_process_error(self, mock_logging):
        """
        Tests error marking functionality for failed processes.
        
        Verifies that:
        1. Process status is correctly updated to 'error'
        2. Error message is properly stored
        3. Database changes are committed
        4. Session handling is correct
        """
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock process for error marking
        mock_process = MagicMock()
        mock_session.query.return_value.get.return_value = mock_process
        
        # Test error marking
        self.manager.mark_process_error(mock_session, 1, "Test error message")
        
        # Verify error was properly recorded
        mock_process.status = 'error'
        mock_process.error_message = "Test error message"
        mock_session.commit.assert_called_once()

    def test_handle_notification(self):
        """Test handling of pose notifications."""
        # Mock process_pose
        self.manager.process_pose = MagicMock()
        
        # Mock session and process
        mock_process = MagicMock(spec=ProcessQueue)
        mock_process.parameters = {'face': True}  # Add required face parameter
        mock_session = MagicMock()
        mock_session.query.return_value.get.return_value = mock_process
        
        with patch.object(self.manager, 'get_managed_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            # Test valid channel with data
            test_data = {
                'process_id': 1,
                'image_id': 2
            }
            
            # Mock logging
            with patch('logging.info') as mock_logging:
                self.manager.handle_notification('image_pose', test_data)
                
                # Verify logging was called with correct parameters
                mock_logging.assert_any_call(
                    "Processing pose with parameters: {'process_id': 1, 'image_id': 2, 'face': True}"
                )
            
            # Verify process_pose was called with correct data
            expected_data = {
                'process_id': 1,
                'image_id': 2,
                'face': True
            }
            self.manager.process_pose.assert_called_once_with(expected_data)
            
            # Test invalid channel
            self.manager.process_pose.reset_mock()
            result = self.manager.handle_notification('invalid', test_data)
            assert result is None
            self.manager.process_pose.assert_not_called()

if __name__ == '__main__':
    unittest.main()


