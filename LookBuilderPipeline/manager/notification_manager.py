import select
import psycopg2
import time
import logging
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
from sqlalchemy import asc

from LookBuilderPipeline.manager.db_manager import DBManager
from LookBuilderPipeline.models.proccess_queue import ProcessQueue


class NotificationManager:
    def __init__(self):
        # Use the DBManager to get session and engine
        self.db_manager = DBManager()
        self.session = self.db_manager.get_session()
        self.conn = None
        self.should_listen = False
        self.setup()

    def listen_for_notifications(self, channel_name, max_notifications=10, timeout=30):
        """
        Listen for notifications on specified channel and process queue when notified.
        
        Args:
            channel_name (str): The notification channel to listen to
            max_notifications (int): Maximum number of notifications to process
            timeout (int): How long to listen for in seconds
        """
        conn = self.db_manager.engine.raw_connection()
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        cursor.execute(f"LISTEN {channel_name};")
        
        logging.info(f"Listening for notifications on channel: {channel_name}")
        start_time = time.time()
        notifications = []
        
        while len(notifications) < max_notifications and time.time() - start_time < timeout:
            if select.select([conn], [], [], 5) == ([], [], []):
                logging.debug("Waiting for notification...")
            else:
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    logging.info(f"Received notification on {channel_name}: {notify.payload}")
                    notifications.append(notify.payload)
                    # Process the queue immediately when notification is received
                    self.process_queue()
        
        return notifications
    
    
    def setup_notification_listener(self):
        logging.debug("Setting up notification listener")
        try:
            self.conn = self.db_manager.engine.raw_connection()
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = self.conn.cursor()
            cursor.execute("LISTEN new_proceess;")
            self.should_listen = True
            logging.debug("Notification listener set up successfully")
        except Exception as e:
            logging.error(f"Error setting up notification listener: {str(e)}")
            self.stop_listening()

    def stop_listening(self):
        logging.debug("Stopping notification listener")
        self.should_listen = False
        if self.conn:
            self.conn.close()
            self.conn = None
        logging.debug("Notification listener stopped")

        
    def setup(self):
        """Initialize the notification manager and process any pending items in queue."""
        logging.info("Setting up NotificationManager")
        self.setup_notification_listener()
        # Process any pending items in the queue on startup
        self.process_queue()
        logging.info("NotificationManager setup complete")

    def pop_process(self):
        """Pop the oldest pending process from the queue and mark it as processing."""
        process = None
        try:
            process = self.session.query(ProcessQueue)\
                .filter(ProcessQueue.status == 'pending')\
                .order_by(asc(ProcessQueue.created_at))\
                .first()
            
            if process:
                process.status = 'processing'
                process.updated_at = datetime.now()
                self.session.commit()
            return process
        except Exception as e:
            self.session.rollback()
            logging.error(f"Error popping process from queue: {str(e)}")
            if process:
                process.status = f'error: {str(e)[:100]}'
                process.updated_at = datetime.now()
            return None

    def process_queue(self):
        """Process all items in the queue until empty."""
        logging.info("Starting queue processing")
        while True:
            process = self.pop_process()
            if not process:
                logging.info("Queue is empty")
                break
            
            try:
                logging.info(f"Processing queue item {process.process_id}")
                # Here you would implement the actual processing logic
                # based on process.next_step and process.parameters
                
                # Mark as completed after successful processing
                process.status = 'completed'
                process.updated_at = datetime.now()
                self.session.commit()
                
            except Exception as e:
                logging.error(f"Error processing queue item {process.process_id}: {str(e)}")
                # Don't handle the error here since pop_process already did
                break

    def listen_for_ping(self, timeout=30):
        """
        Listen for ping notifications and respond with a pong process.
        
        Args:
            timeout (int): How long to listen for in seconds
        """
        notifications = self.listen_for_notifications('ping', timeout=timeout)
        for ping_id in notifications:
            self.create_pong_process(ping_id)

    def create_pong_process(self, ping_id):
        """Create a new process queue item for pong response."""
        try:
            pong_process = ProcessQueue(
                next_step='pong',
                parameters={'ping_id': ping_id},
                status='pending'
            )
            self.session.add(pong_process)
            self.session.commit()
            logging.info(f"Created pong process for ping {ping_id}")
        except Exception as e:
            self.session.rollback()
            logging.error(f"Error creating pong process: {str(e)}")

