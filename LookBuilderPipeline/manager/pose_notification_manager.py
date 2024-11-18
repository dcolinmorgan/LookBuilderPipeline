import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.pose import detect_pose as image_pose
from PIL import Image
from io import BytesIO
from LookBuilderPipeline.models.image_variant import ImageVariant
from LookBuilderPipeline.models.image import Image
import select

class PoseNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing PoseNotificationManager")
        self.channels = ['image_pose']
        logging.info(f"PoseNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'face']

    def handle_notification(self, channel, data):
        """Handle pose notifications."""
        logging.info(f"PoseNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_pose':
            try:
                with self.get_managed_session() as session:
                    try:
                        process = session.query(ProcessQueue).get(data['process_id'])
                        if process and process.parameters:
                            full_data = {
                                'process_id': data['process_id'],
                                'image_id': data['image_id'],
                                'face': process.parameters.get('face')
                            }
                            logging.info(f"Processing pose with parameters: {full_data}")
                            return self.process_pose(full_data, session)
                        else:
                            logging.error(f"Process {data['process_id']} not found or has no parameters")
                            session.rollback()  # Explicitly rollback on error
                    except Exception as e:
                        logging.error(f"Database error in handle_notification: {str(e)}")
                        session.rollback()  # Explicitly rollback on error
                        raise
            except Exception as e:
                logging.error(f"Session error in handle_notification: {str(e)}")
                return None
        return None

    def process_item(self, pose):
        """Process a single pose notification."""
        return self.process_pose({
            'process_id': pose.process_id,
            'image_id': pose.image_id,
            'face': pose.parameters.get('face')
        })

    def process_pose(self, pose_data, session):
        """Process pose with explicit session management."""
        try:
            validated_data = self.validate_process_data(pose_data)
            process_id = validated_data['process_id']
            
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                self.mark_process_error(session, process_id, "Image not found")
                return None

            variant = image.get_or_create_pose_variant(
                session, 
                face=validated_data['face']
            )
            
            if variant:
                return variant.id
            else:
                self.mark_process_error(session, process_id, "Failed to create variant")
                return None
                
        except Exception as e:
            logging.error(f"Error in process_pose: {str(e)}")
            if 'process_id' in locals():
                self.mark_process_error(session, process_id, str(e))
            raise

    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()

    def _listen_for_notifications(self):
        """Override parent method to add more logging"""
        logging.info("Starting pose notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"pose raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info(f"pose notification added to queue: {notify.channel}")
            except Exception as e:
                logging.error(f"Error in pose notification listener: {str(e)}")
                self._handle_connection_error()
