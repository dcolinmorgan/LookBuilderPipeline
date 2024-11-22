import logging
import argparse
import time
from LookBuilderPipeline.manager.ping_notification_manager import PingNotificationManager
from LookBuilderPipeline.manager.resize_notification_manager import ResizeNotificationManager
from LookBuilderPipeline.manager.segment_notification_manager import SegmentNotificationManager
from LookBuilderPipeline.manager.pose_notification_manager import PoseNotificationManager
from LookBuilderPipeline.manager.loadpipe_notification_manager import LoadPipeNotificationManager
from LookBuilderPipeline.manager.runpipe_notification_manager import RunPipeNotificationManager
from LookBuilderPipeline.manager.sdxl_notification_manager import SDXLNotificationManager
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
    elif mode == 'pose':
        nm = PoseNotificationManager()
    elif mode == 'loadpipe':
        nm = LoadPipeNotificationManager()
    elif mode == 'runpipe':
        nm = RunPipeNotificationManager()
    elif mode == 'sdxl':
        nm = SDXLNotificationManager()
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
    parser.add_argument('--mode', choices=['ping', 'resize','segment','pose','loadpipe','runpipe','sdxl'], required=True,
                      help='Mode to run the application in')

    args = parser.parse_args()

    if args.mode == 'ping':
        run_listener('ping')
    elif args.mode == 'resize':
        run_listener('resize')
    elif args.mode == 'segment':
        run_listener('segment')
    elif args.mode == 'pose':
        run_listener('pose')
    elif args.mode == 'loadpipe':
        run_listener('loadpipe')
    elif args.mode == 'runpipe':
        run_listener('runpipe')
    elif args.mode == 'sdxl':
        run_listener('sdxl')

if __name__ == '__main__':
    main()
