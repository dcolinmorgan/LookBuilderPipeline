import threading
from queue import Queue, Empty
import select
import psycopg2
import time
import logging
import json
import os
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError, DBAPIError, DisconnectionError
from psycopg2.errors import OperationalError

from LookBuilderPipeline.manager.db_manager import DBManager
from LookBuilderPipeline.models.proccess_queue import ProcessQueue


class NotificationManager:
    def __init__(self):
        self.db_manager = DBManager()
        self.session = self.db_manager.get_session()
        self.notification_queue = Queue()
        self.should_listen = True
        self.processor_thread = None
        self.listener_thread = None
        self.conn = None
        self.cursor = None
        # Don't call setup in __init__
    
    def process_existing_queue(self):
        """Process any existing unprocessed notifications in the queue."""
        try:
            logging.info("Checking for existing unprocessed notifications...")
            
            with self.db_manager.get_session() as session:
                # Find all pending ping processes
                pending_pings = session.query(ProcessQueue)\
                    .filter_by(next_step='ping', status='pending')\
                    .all()
                
                if not pending_pings:
                    logging.info("No pending notifications found")
                    return
                
                logging.info(f"Found {len(pending_pings)} unprocessed notifications")
                
                # Process each pending ping
                for ping in pending_pings:
                    try:
                        ping_data = {
                            'process_id': ping.process_id,
                            'image_id': ping.image_id
                        }
                        self.process_ping(ping_data)
                        logging.info(f"Processed existing ping {ping.process_id}")
                        
                    except Exception as e:
                        logging.error(f"Error processing existing ping {ping.process_id}: {str(e)}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error processing existing queue: {str(e)}")
            raise

    def setup(self):
        """Set up the notification manager."""
        logging.info("Setting up NotificationManager")
        
        try:
            # Process any existing notifications first
            self.process_existing_queue()
            
            # Set up notification listening
            self.conn = self.db_manager.get_connection()
            self.cursor = self.conn.cursor()
            
            # Listen only for ping notifications
            self.cursor.execute("LISTEN ping;")
            
            # Start the listener thread
            self.listener_thread = threading.Thread(
                target=self._listen_for_notifications,
                name="NotificationListener"
            )
            self.listener_thread.daemon = True
            self.listener_thread.start()
            
            # Start the processor thread
            self.processor_thread = threading.Thread(
                target=self._process_notifications,
                name="NotificationProcessor"
            )
            self.processor_thread.daemon = True
            self.processor_thread.start()
            
            logging.info("NotificationManager setup complete")
            
        except Exception as e:
            logging.error(f"Error during NotificationManager setup: {str(e)}")
            raise

    def _listen_for_notifications(self):
        """Background thread to listen for notifications."""
        logging.info("Starting notification listener thread")
        
        while self.should_listen:
            try:
                logging.debug("Waiting for notifications...")
                if select.select([self.conn], [], [], 1.0)[0]:
                    self.conn.poll()
                    while self.conn.notifies:
                        notify = self.conn.notifies.pop()
                        logging.info(f"Raw notification received: {notify}")
                        logging.info(f"Channel: {notify.channel}, Payload: {notify.payload}")
                        self.notification_queue.put((notify.channel, notify.payload))
                        #should go to proccesss notifications 
                        logging.info("Notification added to queue")
                else:
                    logging.debug("No notifications in this poll cycle")
            except Exception as e:
                logging.error(f"Error in notification listener: {str(e)}")
                # Try to reconnect
                try:
                    logging.info("Attempting to reconnect...")
                    self.conn = psycopg2.connect(
                        dbname=os.getenv('DB_NAME', 'lookbuilderhub_db'),
                        user=os.getenv('DB_USERNAME', 'lookbuilderhub_user'),
                        password=os.getenv('DB_PASSWORD', 'lookbuilderhub_password'),
                        host=os.getenv('DB_HOST', 'localhost')
                    )
                    self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    cur = self.conn.cursor()
                    cur.execute('LISTEN ping;')
                    logging.info("Reconnected successfully")
                except Exception as reconnect_error:
                    logging.error(f"Reconnection failed: {str(reconnect_error)}")
                    time.sleep(5)  # Wait before trying again

    def _process_notifications(self):
        """Background thread to process notifications."""
        logging.info(f"Processor thread {threading.current_thread().name} started")
        
        while self.should_listen:
            try:
                channel, payload = self.notification_queue.get(timeout=1.0)
                logging.info(f"Processing notification: Channel={channel}, Payload={payload}")
                
                # Only process ping notifications
                if channel == 'ping':
                    self.process_ping(payload)
                    
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing notification: {str(e)}")
                logging.error("Full traceback:", exc_info=True)

    def process_ping(self, ping_data):
        """
        Process a ping notification through its stages.
        """
        try:
            if isinstance(ping_data, str):
                ping_data = json.loads(ping_data)
            
            image_id = ping_data.get('image_id')
            ping_process_id = ping_data.get('process_id')
            
            if not image_id or not ping_process_id:
                raise ValueError("Missing required ping data (image_id or process_id)")
                
            try:
                # Stage 1: Mark as processing
                self.update_ping_status(ping_process_id, 'processing')
                
                try:
                    # Stage 2: Create pong
                    pong_process = self.create_pong_process(image_id, ping_process_id)
                    
                    # Stage 3: Mark as completed
                    self.update_ping_status(ping_process_id, 'completed')
                    
                    return pong_process
                    
                except Exception as e:
                    # If pong creation fails, mark ping as error
                    logging.error(f"Error in pong creation: {str(e)}")
                    self.update_ping_status(ping_process_id, 'error')
                    raise
                    
            except Exception as e:
                logging.error(f"Error in process_ping: {str(e)}")
                # Ensure ping is marked as error if not already
                try:
                    self.update_ping_status(ping_process_id, 'error')
                except Exception as update_error:
                    logging.error(f"Failed to update ping status to error: {str(update_error)}")
                raise
                
        except Exception as e:
            logging.error(f"Error processing ping: {str(e)}")
            raise

    def update_ping_status(self, ping_process_id, status):
        """Update the status of a ping process."""
        try:
            # Get a fresh session for this operation
            session = self.db_manager.get_session()
            
            try:
                ping_process = session.query(ProcessQueue)\
                    .filter(ProcessQueue.process_id == ping_process_id)\
                    .with_for_update()\
                    .first()
                
                if not ping_process:
                    raise ValueError(f"Ping process {ping_process_id} not found")
                    
                ping_process.status = status
                ping_process.updated_at = datetime.now()
                session.commit()
                logging.info(f"Updated ping process {ping_process_id} status to {status}")
                
            except Exception as e:
                session.rollback()
                logging.error(f"Error updating ping status: {str(e)}")
                raise
            finally:
                session.close()
                
        except Exception as e:
            logging.error(f"Error in update_ping_status: {str(e)}")
            raise

    def create_pong_process(self, image_id, ping_process_id):
        """Create a new pong process."""
        logging.info(f"Creating pong process for ping {ping_process_id}")
        
        try:
            # Get a fresh session for this operation
            session = self.db_manager.get_session()
            
            pong_process = ProcessQueue(
                image_id=image_id,
                next_step='pong',
                parameters={
                    'ping_process_id': ping_process_id,
                    'image_id': image_id
                },
                status='pending'
            )
            
            session.add(pong_process)
            session.commit()
            
            # Get the ID before closing the session
            pong_id = pong_process.process_id
            logging.info(f"Created pong process with ID: {pong_id}")
            
            return pong_id
            
        except Exception as e:
            session.rollback()
            logging.error(f"Error creating pong process: {str(e)}")
            raise
        finally:
            session.close()

    def stop(self):
        """Stop the notification manager."""
        logging.info("Stopping NotificationManager...")
        self.should_listen = False
        
        # Stop listener thread
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=5)
            if self.listener_thread.is_alive():
                logging.warning("Listener thread did not stop cleanly")
        
        # Stop processor thread
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
            if self.processor_thread.is_alive():
                logging.warning("Processor thread did not stop cleanly")
        
        # Clean up database connections
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        if self.session:
            self.session.close()
            
        logging.info("NotificationManager stopped")

