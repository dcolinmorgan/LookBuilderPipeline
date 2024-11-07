import logging
import argparse
import time
from LookBuilderPipeline.manager.ping_notification_manager import PingNotificationManager
from LookBuilderPipeline.manager.resize_notification_manager import ResizeNotificationManager
from LookBuilderPipeline.manager.segment_notification_manager import SegmentNotificationManager
logging.basicConfig(level=logging.INFO)

def run_listener(mode):
    """Run the ping notification listener."""
    logging.info("Starting ping listener...")
    if mode == 'ping':
        nm = PingNotificationManager()
    elif mode == 'resize':
        nm = ResizeNotificationManager()
    elif mode == 'segment':
        nm = SegmentNotificationManager()
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

def main():
    parser = argparse.ArgumentParser(description='LookBuilder Pipeline')
    parser.add_argument('--mode', choices=['ping', 'resize','segment'], required=True,
                      help='Mode to run the application in')

    args = parser.parse_args()

    if args.mode == 'ping':
        run_listener('ping')
    elif args.mode == 'resize':
        run_listener('resize')
    elif args.mode == 'segment':
        run_listener('segment')

if __name__ == '__main__':
    main()
