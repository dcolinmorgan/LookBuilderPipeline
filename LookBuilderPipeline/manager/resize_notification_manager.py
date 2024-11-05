import logging
from LookBuilderPipeline.manager.notification_manager import NotificationManager
from LookBuilderPipeline.models.process_queue import ProcessQueue
from LookBuilderPipeline.utils.resize import resize_images
from PIL import Image
from io import BytesIO
from LookBuilderPipeline.models.image_variant import ImageVariant
from LookBuilderPipeline.models.image import Image

class ResizeNotificationManager(NotificationManager):
    def __init__(self):
        super().__init__()
        self.channels = ['resize']
        self.required_fields = ['process_id', 'image_id']

    def handle_notification(self, channel, data):
        """Handle resize notifications."""
        if channel == 'resize':
            return self.process_resize(data)
        return None

    def process_item(self, process):
        """Process a resize item from the queue."""
        if not process.parameters or 'size' not in process.parameters:
            raise ValueError("Process missing required size parameter")

        with self.get_managed_session() as session:
            # Get the image
            image = session.query(Image).get(process.image_id)
            if not image:
                raise ValueError(f"Image {process.image_id} not found")

            # Get or create the resize variant
            variant = image.get_or_create_resize_variant(
                session,
                size=process.parameters['size'],
                aspect_ratio=process.parameters.get('aspect_ratio', 1.0),
                square=process.parameters.get('square', False)
            )

            # Update process status
            self.update_process_status(session, process.process_id, 'completed')
            session.commit()

            return variant

    def process_resize(self, resize_data):
        """Process a resize notification through its stages."""
        validated_data = self.validate_process_data(resize_data)
        process_id = validated_data['process_id']
        
        with self.get_managed_session() as session:
            process = session.query(ProcessQueue)\
                .filter_by(process_id=process_id)\
                .first()
            
            if not process or not process.parameters or 'size' not in process.parameters:
                raise ValueError(f"Process {process_id} missing required size parameter")
            
            size = process.parameters['size']
        
        def execute_resize_process(session):
            logging.info(f"Resizing image {validated_data['image_id']} to size {size}")
            # Here you would add the actual resize logic
            
            self.update_process_status(session, process_id, 'completed')
            return process_id
        
        return self.process_with_error_handling(process_id, execute_resize_process) 