import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue

from PIL import Image
from io import BytesIO
from LookBuilderPipeline.models.image_variant import ImageVariant
from LookBuilderPipeline.models.image import Image
import select

class RunPipeNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        logging.info("Initializing RunPipeNotificationManager")
        self.channels = ['runpipe']
        logging.info(f"RunPipeNotificationManager listening on channels: {self.channels}")
        self.required_fields = ['process_id', 'image_id']

    def handle_notification(self, channel, data):
        """Handle runpipe notifications."""
        logging.info(f"RunPipeNotificationManager received: channel={channel}, data={data}")
        if channel == 'runpipe':
            # Get the full process data from ProcessQueue
            with self.get_managed_session() as session:
                process = session.query(ProcessQueue).get(data['process_id'])
                if process and process.parameters:
                    # Merge the notification data with process parameters
                    full_data = {
                        'process_id': data['process_id'],
                        'image_id': data['image_id'],
                        'pipe': process.parameters.get('pipe'),
                    }
                    logging.info(f"Processing runpipe with parameters: {full_data}")
                    # if process.parameters.get('pipe') == 'sdxl':
                    #     from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL
                    #     self.ImageModelSDXL.generate_image()
                    # elif process.parameters.get('pipe') == 'flux':
                    #     from LookBuilderPipeline.image_models.image_model_fl2 import ImageModelFlux
                    #     self.ImageModelFlux.generate_image()
                                                      
                else:
                    logging.error(f"Process {data['process_id']} not found or has no parameters")
            return None
        logging.warning(f"runpipeNotificationManager received unexpected channel: {channel}")
        return None

    def process_item(self, runpipe):
        """Process a single runpipe notification."""
        return self.process_runpipe({
            'process_id': runpipe.process_id,
            'image_id': runpipe.image_id,
            'pipe': runpipe.parameters.get('pipe'),

        })

    def process_runpipe(self, runpipe_data):
        """Process a runpipe notification through its stages."""
        validated_data = self.validate_process_data(runpipe_data)
        process_id = validated_data['process_id']
        
        def execute_runpipe_process(session):
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
                variant = image.get_or_create_runpipe_variant(
                    session,
                    inverse=validated_data['inverse']
                )
                
                if not variant:
                    error_msg = (
                        f"Failed to create runpipe variant for image {validated_data['image_id']}. "
                        f"The runpipe operation completed but returned no variant. "
                        f"This might indicate an issue with the image processing."
                    )
                    logging.error(error_msg)
                    self.mark_process_error(session, process_id, error_msg)
                    return None
                    
                return variant.id
                
            except Exception as e:
                error_msg = (
                    f"Error creating runpipe variant: {str(e)}. "
                    f"This could be due to invalid image data or insufficient system resources."
                )
                logging.error(error_msg)
                self.mark_process_error(session, process_id, error_msg)
                return None
        
        return self.process_with_error_handling(process_id, execute_runpipe_process)

    def mark_process_error(self, session, process_id, error_message):
        """Mark a process as error with an error message."""
        process = session.query(ProcessQueue).get(process_id)
        if process:
            process.status = 'error'
            process.error_message = error_message
            session.commit()

    def _listen_for_notifications(self):
        """Override parent method to add more logging"""
        logging.info("Starting runpipe notification listener thread")
        
        while self.should_listen:
            try:
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"runpipe raw notification received: {notify}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        logging.info(f"runpipe notification added to queue: {notify.channel}")
            except Exception as e:
                logging.error(f"Error in runpipe notification listener: {str(e)}")
                self._handle_connection_error()
