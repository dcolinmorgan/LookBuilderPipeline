import select
import psycopg2
import time
import logging

from LookBuilderPipeline.manager.db_manager import DBManager


class NotificationManager:
    def __init__(self):
        # Use the DBManager to get session and engine
        self.db_manager = DBManager()
        self.session = self.db_manager.get_session()
        self.conn = None
        self.should_listen = False
        self.setup() 

    def listen_for_notifications(self, max_notifications=10, timeout=30):
        conn = self.db_manager.engine.raw_connection()
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        cursor.execute("LISTEN new_image;")
        
        print("Listening for new processes...")
        start_time = time.time()
        notifications = []
        while len(notifications) < max_notifications and time.time() - start_time < timeout:
            if select.select([conn], [], [], 5) == ([], [], []):
                print("Waiting for notification...")
            else:
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    print(f"Got notification: {notify.payload}")
                    self.process_notification(notify.payload)
                    notifications.append(notify.payload)
        
        return notifications
    
    def process_notification(self, payload):
        """
        Processes a notification payload.
        """
        try:
            image_id = int(payload)
            logging.info(f" *****   here **** Processing new image with ID: {image_id}")
            print(f" *****   here **** Processing new image with ID: {image_id}")
            # Actually process the image
            self.process_image(image_id)
            return True
        except ValueError as e:
            logging.error(f"Invalid image ID in payload: {payload}")
            return False
        except Exception as e:
            logging.error(f"Error processing notification for image {payload}: {str(e)}")
            return False

