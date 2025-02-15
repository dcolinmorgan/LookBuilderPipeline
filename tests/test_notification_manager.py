import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.manager.ping_notification_manager import PingNotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue

@pytest.fixture
def notification_manager():
    """Base notification manager for testing core functionality."""
    return NotificationManager()

@pytest.fixture
def ping_notification_manager():
    """Ping-specific notification manager for testing ping functionality."""
    return PingNotificationManager()

def test_update_ping_status(ping_notification_manager):
    """Test updating a ping process status."""
    # Create a test ping process
    ping_process = ping_notification_manager.create_process(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    # Add process using context manager
    with ping_notification_manager.get_managed_session() as session:
        session.add(ping_process)
        session.flush()
        process_id = ping_process.process_id
    
    # Test status transitions
    with ping_notification_manager.get_managed_session() as session:
        ping_notification_manager.update_process_status(session, process_id, 'processing')
    
    # Verify processing status with fresh session
    with ping_notification_manager.get_managed_session() as session:
        updated_ping = session.query(ProcessQueue)\
            .filter_by(process_id=process_id)\
            .first()
        assert updated_ping.status == 'processing'
    
    # Test next transition
    with ping_notification_manager.get_managed_session() as session:
        ping_notification_manager.update_process_status(session, process_id, 'completed')
    
    # Verify completed status with fresh session
    with ping_notification_manager.get_managed_session() as session:
        updated_ping = session.query(ProcessQueue)\
            .filter_by(process_id=process_id)\
            .first()
        assert updated_ping.status == 'completed'

def test_create_pong_process(ping_notification_manager):
    """Test creating a pong process."""
    # Create initial ping process
    ping_process = ping_notification_manager.create_process(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with ping_notification_manager.get_managed_session() as session:
        session.add(ping_process)
        session.flush()
        ping_id = ping_process.process_id
    
        # Create pong process via execute_ping_process within process_ping
        pong_id = ping_notification_manager.process_ping(
            ping_data={
                'image_id': 1,
                'process_id': ping_id
            }
        )
    
        # Verify pong process - use same session
        pong = session.query(ProcessQueue)\
            .filter_by(process_id=pong_id)\
            .first()
        assert pong is not None
        assert pong.next_step == 'pong'
        assert pong.status == 'pending'
        assert pong.image_id == 1
        assert pong.parameters['ping_process_id'] == ping_id

def test_full_ping_processing(ping_notification_manager):
    """Test the complete ping-to-pong process."""
    # Create initial ping process
    ping_process = ProcessQueue(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with ping_notification_manager.get_managed_session() as session:
        session.add(ping_process)
        session.flush()
        ping_id = ping_process.process_id
    
    # Process the ping
    ping_data = {
        'process_id': ping_id,
        'image_id': 1
    }
    ping_notification_manager.process_ping(ping_data)
    
    # Verify state transitions
    with ping_notification_manager.get_managed_session() as session:
        # Check ping status progression
        ping = session.query(ProcessQueue)\
            .filter_by(process_id=ping_id)\
            .first()
        assert ping.status == 'completed'
        
        # Verify pong was created
        pong = session.query(ProcessQueue)\
            .filter_by(next_step='pong')\
            .filter_by(parameters={'ping_process_id': ping_id, 'image_id': 1})\
            .first()
        assert pong is not None
        assert pong.status == 'pending'

def test_error_handling(ping_notification_manager):
    """Test error handling during ping processing."""
    # Create ping process
    ping_process = ping_notification_manager.create_process(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with ping_notification_manager.get_managed_session() as session:
        session.add(ping_process)
        session.flush()
        ping_id = ping_process.process_id
    
    ping_data = {
        'process_id': ping_id,
        'image_id': 1
    }
    
    # Simulate error during pong creation
    def raise_error(*args, **kwargs):
        raise Exception('Test error')
    
    with patch.object(ping_notification_manager, 'create_process', side_effect=raise_error):
        with pytest.raises(Exception):
            ping_notification_manager.process_ping(ping_data)
        
        # Verify error state
        with ping_notification_manager.get_managed_session() as session:
            ping = session.query(ProcessQueue)\
                .filter_by(process_id=ping_id)\
                .first()
            assert ping.status == 'error'

def test_process_existing_queue(ping_notification_manager):
    """Test processing of existing unprocessed notifications on startup."""
    # Set lower max attempts for testing
    ping_notification_manager.max_processing_attempts = 3
    ping_notification_manager.process_delay = 0.01  # Faster processing for tests
    
    # Create multiple unprocessed ping processes
    unprocessed_pings = []
    for i in range(3):
        ping_process = ping_notification_manager.create_process(
            image_id=1,
            next_step='ping',
            status='pending'
        )
        with ping_notification_manager.get_managed_session() as session:
            session.add(ping_process)
            session.flush()
            unprocessed_pings.append(ping_process.process_id)
    
    # Process existing queue
    ping_notification_manager.process_existing_queue()
    
    # Verify all unprocessed pings were processed
    with ping_notification_manager.get_managed_session() as session:
        for ping_id in unprocessed_pings:
            ping = session.query(ProcessQueue)\
                .filter_by(process_id=ping_id)\
                .first()
            assert ping.status == 'completed'
            
            # Verify pong was created for each ping
            pong = session.query(ProcessQueue)\
                .filter_by(next_step='pong')\
                .filter(ProcessQueue.parameters['ping_process_id'].astext == str(ping_id))\
                .first()
            assert pong is not None
            assert pong.status == 'pending'

def test_notification_channels(ping_notification_manager):
    """Test that ping notification manager has correct channels configured."""
    assert ping_notification_manager.channels == ['ping']
    assert 'pong' not in ping_notification_manager.channels

def test_process_notification(ping_notification_manager):
    """Test that process_notification correctly handles ping channel."""
    with patch.object(ping_notification_manager, 'process_ping') as mock_process_ping:
        # Test ping channel
        ping_notification_manager.process_notification('ping', '{"process_id": 1, "image_id": 1}')
        mock_process_ping.assert_called_once()
        
        # Test other channel (should not process)
        mock_process_ping.reset_mock()
        ping_notification_manager.process_notification('other', '{"process_id": 1, "image_id": 1}')
        mock_process_ping.assert_not_called()
