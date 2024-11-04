import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.proccess_queue import ProcessQueue

@pytest.fixture
def mock_db_setup():
    with patch('LookBuilderPipeline.manager.notification_manager.DBManager') as mock_db_manager:
        # Mock DBManager and its connection
        mock_engine = Mock()
        mock_conn = Mock()
        mock_session = Mock()
        
        # Setup the mock connection
        mock_engine.raw_connection.return_value = mock_conn
        mock_db_instance = Mock()
        mock_db_instance.engine = mock_engine
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance
        
        yield mock_db_instance, mock_conn, mock_session

def test_startup_processes_existing_queue(mock_db_setup):
    """Test that startup processes existing items in queue"""
    mock_db_instance, mock_conn, mock_session = mock_db_setup
    
    # Create mock queue items
    mock_items = [
        ProcessQueue(
            process_id=i,
            status='pending',
            created_at=datetime.now()
        ) for i in range(1, 4)
    ]
    
    # Setup mock query responses
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.side_effect = mock_items + [None]
    
    # Initialize and run setup
    nm = NotificationManager()
    
    # Verify all items were processed
    assert mock_session.commit.call_count == 6  # 3 items × 2 commits (status update + completion)
    
def test_notification_triggers_queue_processing(mock_db_setup):
    """Test that receiving a notification triggers queue processing"""
    mock_db_instance, mock_conn, mock_session = mock_db_setup
    
    # Create mock notification
    mock_notify = Mock()
    mock_notify.payload = "123"
    mock_conn.notifies = [mock_notify]
    
    # Setup mock process queue item
    mock_process = ProcessQueue(process_id=1, status='pending', created_at=datetime.now())
    
    # Need to handle both the initial setup call and the notification processing
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.side_effect = [
        None,  # For initial setup
        mock_process,  # For notification processing
        None  # To end processing
    ]
    
    with patch('LookBuilderPipeline.manager.notification_manager.select.select', 
              return_value=([mock_conn], [], [])):
        nm = NotificationManager()
        notifications = nm.listen_for_notifications('test_channel', max_notifications=1, timeout=1)
        
        # Verify notification was received and queue was processed
        assert len(notifications) == 1
        assert notifications[0] == "123"
        assert mock_conn.cursor.called
        assert mock_session.commit.call_count == 2  # One for status update, one for completion

def test_process_queue_handles_error(mock_db_setup):
    """Test that process queue handles errors appropriately"""
    mock_db_instance, mock_conn, mock_session = mock_db_setup
    
    # Setup mock process that will raise an exception
    mock_process = ProcessQueue(
        process_id=1,
        status='pending',
        created_at=datetime.now()
    )
    
    # Setup the mock to return our process once, then None
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.side_effect = [mock_process, None]
    
    # Make first commit raise an exception
    error_message = "Database connection lost"
    mock_session.commit.side_effect = Exception(error_message)
    
    nm = NotificationManager()
    nm.process_queue()
    
    # Verify error handling
    assert mock_process.status.startswith('error: Database connection lost')
    mock_session.rollback.assert_called_once()
    assert mock_session.commit.call_count == 1  # Only o ne commit attempt

def test_ping_creates_pong_process(mock_db_setup):
    """Test that receiving a ping notification creates a pong process."""
    mock_db_instance, mock_conn, mock_session = mock_db_setup
    
    # Create mock ping notification
    mock_notify = Mock()
    mock_notify.payload = "ping-123"
    mock_conn.notifies = [mock_notify]
    
    # Setup mock process queue item for initial process_queue call
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
    
    with patch('LookBuilderPipeline.manager.notification_manager.select.select', 
              return_value=([mock_conn], [], [])):
        nm = NotificationManager()
        nm.listen_for_ping(timeout=1)
        
        # Verify pong process was created
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # Get the ProcessQueue object that was added
        added_process = mock_session.add.call_args[0][0]
        assert isinstance(added_process, ProcessQueue)
        assert added_process.next_step == 'pong'
        assert added_process.parameters['ping_id'] == 'ping-123'
        assert added_process.status == 'pending'