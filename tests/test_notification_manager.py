import pytest
from unittest.mock import Mock, patch
from LookBuilderPipeline.manager.notification_manager import NotificationManager

def test_listen_for_notifications():
    # Create a mock notification object
    mock_notify = Mock()
    mock_notify.payload = "123"  # Sample image ID

    # Create a mock connection
    mock_conn = Mock()
    mock_conn.notifies = [mock_notify]  # Add our mock notification to the connection
    
    with patch('LookBuilderPipeline.manager.notification_manager.select.select', 
               return_value=([mock_conn], [], [])), \
         patch.object(NotificationManager, 'process_notification', return_value=True):
        
        # Initialize notification manager
        nm = NotificationManager()
        
        # Mock the database connection
        nm.db_manager.engine.raw_connection = Mock(return_value=mock_conn)
        
        # Test the method
        notifications = nm.listen_for_notifications(max_notifications=1, timeout=1)
        
        # Assertions
        assert len(notifications) == 1
        assert notifications[0] == "123"
        assert mock_conn.set_isolation_level.called
        assert mock_conn.cursor.called
        nm.process_notification.assert_called_once_with("123") 