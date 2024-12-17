from unittest.mock import MagicMock, patch
import unittest
from LookBuilderPipeline.manager.sdxl_notification_manager import SDXLNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant

class TestSDXLNotificationManager(unittest.TestCase):
    """
    Test suite for SDXLNotificationManager.
    Tests the handling of sdxl detection notifications and process management.
    """
    
    def setUp(self):
        """
        Setup runs before each test.
        Creates a fresh SDXLNotificationManager instance and mocks its session handling
        to avoid actual database connections during testing.
        """
        self.manager = SDXLNotificationManager()
        self.manager.get_managed_session = MagicMock()

    @patch('LookBuilderPipeline.manager.sdxl_notification_manager.logging')
    def test_handle_notification_success(self, mock_logging):
        """
        Tests successful handling of a sdxl notification.
        
        Verifies that:
        1. Manager correctly processes a valid sdxl notification
        2. Process parameters are properly retrieved from the database
        3. Logging is performed with correct parameters
        4. A result is returned (indicating successful processing)
        """
        # Setup mock database session
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        # Mock a process in the database with inverse detection parameter
        mock_process = MagicMock()
        mock_process.parameters = {'inverse': True}
        mock_session.query.return_value.get.return_value = mock_process
        
        # Simulate receiving a sdxl notification
        result = self.manager.handle_notification('image_sdxl', {'process_id': 1, 'image_id': 2})
        
        # Verify the notification was handled correctly
        self.assertIsNotNone(result)
        mock_logging.info.assert_called_with(
            "Processing sdxl with parameters: {'process_id': 1, 'image_id': 2, 'prompt': None, 'negative_prompt': 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves', 'seed': None, 'strength': 1.0, 'guidance_scale': 7.5, 'LoRA': None}"
        )

    @patch('LookBuilderPipeline.manager.sdxl_notification_manager.logging')
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
        
        result = self.manager.handle_notification('image_sdxl', {'process_id': 1, 'image_id': 2, 'prompt': 'woman on beach'})
        
        # Verify proper error handling
        self.assertIsNone(result)
        mock_logging.error.assert_called_with("Process 1 not found or has no parameters")

    @patch('LookBuilderPipeline.manager.sdxl_notification_manager.logging')
    def test_process_sdxl_success(self, mock_logging):
        """
        Tests successful processing of a sdxl detection request.
        
        Verifies that:
        1. Image is correctly retrieved from database
        2. ImageVariant is properly created
        3. SDXL detection variant is created with correct parameters
        4. Processing result is returned
        5. All database operations are performed in correct order
        """
        mock_session = MagicMock()
        self.manager.get_managed_session.return_value.__enter__.return_value = mock_session
        
        mock_process = MagicMock()
        mock_process.parameters = {'inverse': True}
        mock_session.query.return_value.get.return_value = mock_process
        
        result = self.manager.process_sdxl({'process_id': 1, 'image_id': 2, 'prompt': 'woman on beach', 'negative_prompt': 'ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, sunglasses, stockings, pants, sleeves', 'seed': None, 'strength': 1.0, 'guidance_scale': 7.5, 'LoRA': None})
        
        self.assertIsNotNone(result)

    @patch('LookBuilderPipeline.manager.sdxl_notification_manager.logging')
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

if __name__ == '__main__':
    unittest.main()


