import logging
import argparse
import time
from LookBuilderPipeline.manager.ping_notification_manager import PingNotificationManager
from LookBuilderPipeline.manager.resize_notification_manager import ResizeNotificationManager

logging.basicConfig(level=logging.INFO)

def run_ping_listener():
    """Run the ping notification listener."""
    logging.info("Starting ping listener...")
    nm = PingNotificationManager()
    nm.setup()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        nm.stop()
    except Exception as e:
        logging.error(f"Error in ping listener: {str(e)}")
        nm.stop()
        raise

def run_resize_listener():
    """Run the resize notification listener."""
    logging.info("Starting resize listener...")
    try:
        nm = ResizeNotificationManager()
        nm.setup()
        nm.process_existing_queue()
        nm.listen()
    except Exception as e:
        logging.error(f"Error during resize listener setup: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='LookBuilder Pipeline')
    parser.add_argument('--mode', choices=['ping', 'resize'], required=True,
                      help='Mode to run the application in')

    args = parser.parse_args()

    if args.mode == 'ping':
        run_ping_listener()
    elif args.mode == 'resize':
        run_resize_listener()

if __name__ == '__main__':
    main()
