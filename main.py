import logging
import argparse
import time
import signal
import os
from LookBuilderPipeline.manager.ping_notification_manager import PingNotificationManager
from LookBuilderPipeline.manager.resize_notification_manager import ResizeNotificationManager
from LookBuilderPipeline.manager.segment_notification_manager import SegmentNotificationManager
from LookBuilderPipeline.manager.pose_notification_manager import PoseNotificationManager
from LookBuilderPipeline.manager.gen_notification_manager import GenNotificationManager

logging.basicConfig(level=logging.INFO)

def handle_sigtstp(signum, frame):
    """Handle Ctrl+Z (SIGTSTP) by killing the process."""
    logging.info("Received SIGTSTP (Ctrl+Z). Killing process...")
    os._exit(0)

def run_listener(mode):
    """Run the notification listener with SIGTSTP handling."""
    logging.info(f"Starting {mode} listener...")
    
    # Register SIGTSTP handler
    signal.signal(signal.SIGTSTP, handle_sigtstp)
    
    managers = {
        'ping': PingNotificationManager,
        'resize': ResizeNotificationManager,
        'segment': SegmentNotificationManager,
        'pose': PoseNotificationManager,
        'image_gen': GenNotificationManager
    }
    
    nm = managers[mode]()
    nm.setup()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        nm.stop()
        os._exit(0)

def main():
    parser = argparse.ArgumentParser(description='LookBuilder Pipeline')
    parser.add_argument(
        '--mode', 
        choices=['ping', 'resize', 'segment', 'pose', 'image_gen'],
        required=True,
        help='Mode to run the application in'
    )

    args = parser.parse_args()
    run_listener(args.mode)

if __name__ == '__main__':
    main()
    os.execv(sys.executable, ['python'] + sys.argv)
