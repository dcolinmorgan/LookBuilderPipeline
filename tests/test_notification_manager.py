import pytest
from unittest.mock import Mock, patch
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.manager.db_manager import DBManager
from LookBuilderPipeline.config import Config

@patch('LookBuilderPipeline.manager.notification_manager.DBManager')
def test_listen_for_notifications(mock_db_manager):
    # Mock DBManager and its connection
    mock_engine = Mock()
    mock_conn = Mock()
    mock_conn.notifies = []
    
    # Setup the mock connection
    mock_engine.raw_connection.return_value = mock_conn
    mock_db_instance = Mock()
    mock_db_instance.engine = mock_engine
    mock_db_instance.get_session.return_value = Mock()  # Mock the session
    mock_db_manager.return_value = mock_db_instance

    # Create a mock notification
    mock_notify = Mock()
    mock_notify.payload = "123"
    mock_conn.notifies = [mock_notify]

    with patch('LookBuilderPipeline.manager.notification_manager.select.select', 
               return_value=([mock_conn], [], [])), \
         patch.object(NotificationManager, 'process_notification', return_value=True):
        
        # Initialize notification manager
        nm = NotificationManager()
        
        # Test the method
        notifications = nm.listen_for_notifications(max_notifications=1, timeout=1)
        
        # Assertions
        assert len(notifications) == 1
        assert notifications[0] == "123"
        assert mock_conn.set_isolation_level.called
        assert mock_conn.cursor.called
        nm.process_notification.assert_called_once_with("123")