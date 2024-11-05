import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.proccess_queue import ProcessQueue

@pytest.fixture
def notification_manager():
    return NotificationManager()

def test_update_ping_status(notification_manager):
    """Test updating a ping process status."""
    # Create a test ping process
    ping_process = ProcessQueue(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with notification_manager.session.begin():
        notification_manager.session.add(ping_process)
        notification_manager.session.flush()
        process_id = ping_process.process_id
    
    # Test status transitions
    notification_manager.update_ping_status(process_id, 'processing')
    
    # Verify processing status with fresh session
    with notification_manager.db_manager.get_session() as verify_session:
        updated_ping = verify_session.query(ProcessQueue)\
            .filter_by(process_id=process_id)\
            .first()
        assert updated_ping.status == 'processing'
    
    # Test next transition
    notification_manager.update_ping_status(process_id, 'completed')
    
    # Verify completed status with fresh session
    with notification_manager.db_manager.get_session() as verify_session:
        updated_ping = verify_session.query(ProcessQueue)\
            .filter_by(process_id=process_id)\
            .first()
        assert updated_ping.status == 'completed'

def test_create_pong_process(notification_manager):
    """Test creating a pong process."""
    # Create initial ping process
    ping_process = ProcessQueue(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with notification_manager.session.begin():
        notification_manager.session.add(ping_process)
        notification_manager.session.flush()
        ping_id = ping_process.process_id
    
    # Create pong process
    pong_id = notification_manager.create_pong_process(
        image_id=1,
        ping_process_id=ping_id
    )
    
    # Verify pong process
    with notification_manager.session.begin():
        pong = notification_manager.session.query(ProcessQueue)\
            .filter_by(process_id=pong_id)\
            .first()
        assert pong.next_step == 'pong'
        assert pong.status == 'pending'
        assert pong.image_id == 1
        assert pong.parameters['ping_process_id'] == ping_id

def test_full_ping_processing(notification_manager):
    """Test the complete ping-to-pong process."""
    # Create initial ping process
    ping_process = ProcessQueue(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with notification_manager.session.begin():
        notification_manager.session.add(ping_process)
        notification_manager.session.flush()
        ping_id = ping_process.process_id
    
    # Process the ping
    ping_data = {
        'process_id': ping_id,
        'image_id': 1
    }
    notification_manager.process_ping(ping_data)
    
    # Verify state transitions
    with notification_manager.session.begin():
        # Check ping status progression
        ping = notification_manager.session.query(ProcessQueue)\
            .filter_by(process_id=ping_id)\
            .first()
        assert ping.status == 'completed'
        
        # Verify pong was created
        pong = notification_manager.session.query(ProcessQueue)\
            .filter_by(next_step='pong')\
            .filter_by(parameters={'ping_process_id': ping_id, 'image_id': 1})\
            .first()
        assert pong is not None
        assert pong.status == 'pending'

def test_error_handling(notification_manager):
    """Test error handling during ping processing."""
    # Create ping process
    ping_process = ProcessQueue(
        image_id=1,
        next_step='ping',
        status='pending'
    )
    
    with notification_manager.session.begin():
        notification_manager.session.add(ping_process)
        notification_manager.session.flush()
        ping_id = ping_process.process_id
    
    ping_data = {
        'process_id': ping_id,
        'image_id': 1
    }
    
    # Simulate error during pong creation
    with patch.object(notification_manager, 'create_pong_process', side_effect=Exception('Test error')):
        with pytest.raises(Exception):
            notification_manager.process_ping(ping_data)
        
        # Verify error state
        with notification_manager.session.begin():
            ping = notification_manager.session.query(ProcessQueue)\
                .filter_by(process_id=ping_id)\
                .first()
            assert ping.status == 'error'

def test_process_existing_queue(notification_manager):
    """Test processing of existing unprocessed notifications on startup."""
    # Create multiple unprocessed ping processes
    unprocessed_pings = []
    for i in range(3):
        ping_process = ProcessQueue(
            image_id=1,
            next_step='ping',
            status='pending'
        )
        with notification_manager.session.begin():
            notification_manager.session.add(ping_process)
            notification_manager.session.flush()
            unprocessed_pings.append(ping_process.process_id)
    
    # Create a completed ping to verify we only process pending ones
    completed_ping = ProcessQueue(
        image_id=1,
        next_step='ping',
        status='completed'
    )
    with notification_manager.session.begin():
        notification_manager.session.add(completed_ping)
        notification_manager.session.flush()
        completed_ping_id = completed_ping.process_id
    
    # Process existing queue
    notification_manager.process_existing_queue()
    
    # Verify all unprocessed pings were processed
    with notification_manager.db_manager.get_session() as verify_session:
        # Check all original pings are completed
        for ping_id in unprocessed_pings:
            ping = verify_session.query(ProcessQueue)\
                .filter_by(process_id=ping_id)\
                .first()
            assert ping.status == 'completed'
            
            # Verify pong was created for each ping
            pong = verify_session.query(ProcessQueue)\
                .filter_by(next_step='pong')\
                .filter(ProcessQueue.parameters['ping_process_id'].astext == str(ping_id))\
                .first()
            assert pong is not None
            assert pong.status == 'pending'
        
        # Verify the already completed ping wasn't reprocessed
        completed = verify_session.query(ProcessQueue)\
            .filter_by(process_id=completed_ping_id)\
            .first()
        assert completed.status == 'completed'
        
        # Verify no extra pongs were created
        pong_count = verify_session.query(ProcessQueue)\
            .filter_by(next_step='pong')\
            .count()
        assert pong_count == len(unprocessed_pings)
    
    # Create a pending pong to verify we don't process pongs
    pending_pong = ProcessQueue(
        image_id=1,
        next_step='pong',
        status='pending'
    )
    with notification_manager.session.begin():
        notification_manager.session.add(pending_pong)
        notification_manager.session.flush()
        pending_pong_id = pending_pong.process_id
    
    # Additional verification
    pong = verify_session.query(ProcessQueue)\
        .filter_by(process_id=pending_pong_id)\
        .first()
    assert pong.status == 'pending'
    assert pong.next_step == 'pong'