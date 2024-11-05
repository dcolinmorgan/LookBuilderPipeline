from datetime import datetime
import logging
import json
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.proccess_queue import ProcessQueue

class PingNotificationManager(NotificationManager):
    """Handles ping-specific notification processing."""
    def __init__(self):
        super().__init__()
        self.channels = ['ping']
        self.required_fields = ['process_id', 'image_id']  # Define required fields
    
    def process_notification(self, channel, payload):
        """Process notifications for the ping channel."""
        if channel == 'ping':
            parsed_data = self.parse_payload(payload)
            validated_data = self.validate_process_data(parsed_data)
            self.process_ping(validated_data)
    
    def process_existing_queue(self):
        """Process any existing unprocessed ping notifications."""
        try:
            logging.info("Checking for existing unprocessed ping notifications...")
            
            with self.get_managed_session() as session:
                pending_pings = self.get_processes_for_client(
                    session,
                    next_step='ping',
                    status='pending'
                ).all()
                
                if not pending_pings:
                    logging.info("No pending ping notifications found")
                    return
                
                logging.info(f"Found {len(pending_pings)} unprocessed ping notifications")
                
                for ping in pending_pings:
                    try:
                        self.process_ping({
                            'process_id': ping.process_id,
                            'image_id': ping.image_id
                        })
                        logging.info(f"Processed existing ping {ping.process_id}")
                    except Exception as e:
                        logging.error(f"Error processing existing ping {ping.process_id}: {str(e)}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error processing existing queue: {str(e)}")
            raise

    def process_ping(self, ping_data):
        """Process a ping notification through its stages."""
        validated_data = self.validate_process_data(ping_data)
        process_id = validated_data['process_id']
        
        def execute_ping_process(session):
            pong_process = self.create_process(
                image_id=validated_data['image_id'],
                next_step='pong',
                parameters={
                    'ping_process_id': process_id,
                    'image_id': validated_data['image_id']
                },
                status='pending'
            )
            session.add(pong_process)
            return pong_process.process_id
        
        return self.process_with_error_handling(process_id, execute_ping_process)

    def create_pong_process(self, image_id, ping_process_id):
        """Create a new pong process."""
        logging.info(f"Creating pong process for ping {ping_process_id}")
        
        with self.get_managed_session() as session:
            pong_process = self.create_process(
                image_id=image_id,
                next_step='pong',
                parameters={
                    'ping_process_id': ping_process_id,
                    'image_id': image_id
                },
                status='pending'
            )
            session.add(pong_process)
            session.flush()  # Ensure we have the ID before committing
            pong_id = pong_process.process_id
            return pong_id