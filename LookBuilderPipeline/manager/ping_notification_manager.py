from datetime import datetime
import logging
import json
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue

class PingNotificationManager(NotificationManager):
    """Handles ping-specific notification processing."""
    def __init__(self):
        super().__init__()
        self.channels = ['ping']
        self.required_fields = ['process_id', 'image_id']  # Define required fields
    
    def process_item(self, ping):
        """Process a single ping notification."""
        return self.process_ping({
            'process_id': ping.process_id,
            'image_id': ping.image_id
        })

    def process_ping(self, ping_data):
        """Process a ping notification through its stages."""
        logging.info(f"Processing ping notification: {ping_data}")
        try:
            validated_data = self.validate_process_data(ping_data)
            process_id = validated_data['process_id']
            
            def execute_ping_process(session):
                logging.debug(f"Creating pong process for ping {process_id}")
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
        except Exception as e:
            logging.error(f"Error processing ping: {str(e)}", exc_info=True)
            raise

    def handle_notification(self, channel, data):
        """Handle ping notifications."""
        logging.debug(f"PingNotificationManager received: channel={channel}, data={data}")
        if channel == 'ping':
            return self.process_ping(data)
        logging.warning(f"PingNotificationManager received unexpected channel: {channel}")
        return None
