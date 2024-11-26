import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.models.image import Image
import select

class SegmentNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing SegmentNotificationManager")
        self.channels = ['image_segment']
        logging.info(f"SegmentNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'inverse']

    def handle_notification(self, channel, data):
        """Handle segment notifications."""
        logging.info(f"SegmentNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_segment':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'inverse': process.parameters.get('inverse')
                    }
                    logging.info(f"Processing segment with parameters: {full_data}")
                    return self.process_segment(full_data)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"SegmentNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, segment):
        """Process a single segment notification."""
        return self.process_segment({
            'process_id': segment.process_id,
            'image_id': segment.image_id,
            'inverse': segment.parameters.get('inverse')
        })

    def process_segment(self, segment_data):
        """Process a segment notification through its stages."""
        validated_data = self.validate_process_data(segment_data)
        process_id = validated_data['process_id']
        
        def execute_segment_process(session):
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
                self.session = session
                variant = image.get_or_create_segment_variant(
                    session=self.session,
                    inverse=validated_data['inverse']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create segment variant for image {validated_data['image_id']}. "
                        f"The segment operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                return variant.id
                
            except Exception as e:
                error_msg = (
                    f"Error creating segment variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_segment_process)

    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()

    def _listen_for_notifications(self):
        """Override parent method to add more logging"""
        logging.info("Starting segment notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"segment raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info(f"segment notification added to queue: {notify.channel}")
            except Exception as e:
                logging.error(f"Error in segment notification listener: {str(e)}")
                self._handle_connection_error()
