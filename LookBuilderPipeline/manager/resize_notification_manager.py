import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.utils.resize import resize_images
from PIL import Image
from io import BytesIO
from LookBuilderPipeline.models.image_variant import ImageVariant
from LookBuilderPipeline.models.image import Image
import select

class ResizeNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing ResizeNotificationManager")
        self.channels = ['image_resize']
        logging.info(f"ResizeNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id', 'size']

    def handle_notification(self, channel, data):
        """Handle resize notifications."""
        logging.info(f"ResizeNotificationManager received: channel={channel}, data={data}")
        if channel == 'image_resize':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    # full_data = {
                    #     'process_id': data['process_id'],
                    #     'image_id': data['image_id'],
                    #     'size': process.parameters.get('size'),
                    #     'aspect_ratio': process.parameters.get('aspect_ratio', 1.0),
                    #     'square': process.parameters.get('square', False)
                    # }
                    # logging.info(f"Processing resize with parameters: {size}")
                    return self.process_item(process)
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"ResizeNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, resize):
        """Process a single resize notification."""
        # Validate size parameter first
        if not resize.parameters or 'size' not in resize.parameters:
            raise ValueError("Process is missing required size parameter")
        
        # Create validated data from resize object
        validated_data = {
            'process_id': resize.process_id,
            'image_id': resize.image_id,
            'size': resize.parameters.get('size'),
            'aspect_ratio': resize.parameters.get('aspect_ratio', 1.0),
            'square': resize.parameters.get('square', False)
        }
        
        process_id = validated_data['process_id']
        
        def execute_resize_process(session):
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
                variant = image.get_or_create_resize_variant(
                    session,
                    size=validated_data['size'],
                    aspect_ratio=validated_data.get('aspect_ratio', 1.0),
                    square=validated_data.get('square', False)
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create resize variant for image {validated_data['image_id']}. "
                        f"The resize operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                return variant.id
                
            except Exception as e:
                error_msg = (
                    f"Error creating resize variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_resize_process)

    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()

    def _listen_for_notifications(self):
        """Override parent method to add more logging"""
        logging.info("Starting resize notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"Resize raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info(f"Resize notification added to queue: {notify.channel}")
            except Exception as e:
                logging.error(f"Error in resize notification listener: {str(e)}")
                self._handle_connection_error()
