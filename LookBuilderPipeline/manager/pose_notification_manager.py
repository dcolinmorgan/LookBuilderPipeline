import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
from LookBuilderPipeline.models.image_variant import ImageVariant
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
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'face': process.parameters.get('face')
                    }
                    logging.info(f"Processing pose with parameters: {full_data}")
                    return self.process_pose(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"PoseNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, pose):
        """Process a single pose notification."""
        return self.process_pose({
            'process_id': pose.process_id,
            'image_id': pose.image_id,
            'face': pose.parameters.get('face')
        })

    def process_pose(self, pose_data):
        """Process a pose notification through its stages."""
        validated_data = self.validate_process_data(pose_data)
        process_id = validated_data['process_id']
        
        def execute_pose_process(session):
            # Get the image
            image = session.query(Image).get(validated_data['image_id'])
            if not image:
                error_msg = (
                    f"Image {validated_data['image_id']} not found in database. "
                    f"Please ensure the image was properly uploaded and exists in the database."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None

            # Check if image has data
            image_data = image.get_image_data(session)
            if not image_data:
                error_msg = (
                    f"Image {validated_data['image_id']} exists but has no data. "
                    f"This could be due to an incomplete upload or data corruption. "
                    f"Try re-uploading the image."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None

            try:
                # Create a temporary ImageVariant instance to use get_or_create_variant
                base_variant = ImageVariant(
                    source_image_id=image.image_id,
                    variant_type='pose'
                )
                session.add(base_variant)
                session.flush()
                
                # Now use the source_image_id instead of image_id
                variant = base_variant.get_or_create_variant(
                    session=session,
                    variant_type='pose',
                    face=validated_data['face']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create pose variant for image {validated_data['image_id']}. "
                        f"The pose operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                session.expunge(base_variant)
                return variant
                
            except Exception as e:
                error_msg = (
                    f"Error creating pose variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_pose_process)

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
