import logging
import argparse
import time
from LookBuilderPipeline.manager.notification_manager import NotificationManager

logging.basicConfig(level=logging.INFO)

def run_ping_listener():
    """Run the ping notification listener."""
    logging.info("Starting ping listener...")
    nm = NotificationManager()
    
    try:
        # Keep the main thread alive
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
    parser.add_argument('--mode', choices=['ping'], required=True,
                      help='Mode to run the pipeline in')

    args = parser.parse_args()

    if args.mode == 'ping':
        run_ping_listener()

if __name__ == '__main__':
    main()
