import logging
import signal
import sys
import argparse

def run_ping_listener():
    """
    Launch the notification manager in ping-listening mode.
    """
    from LookBuilderPipeline.manager.notification_manager import NotificationManager
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    nm = NotificationManager()
    
    def signal_handler(sig, frame):
        logger.info("Shutting down ping listener...")
        nm.stop_listening()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting ping listener...")
        while True:
            nm.listen_for_ping(timeout=300)  # 5 minutes timeout
    except Exception as e:
        logger.error(f"Error in ping listener: {str(e)}")
        nm.stop_listening()
        sys.exit(1)

def run_pipeline():
    """Run the main image processing pipeline."""
    from segmentation.segmentation import segment_image
    from pose.pose import detect_pose
    from image_models.image_model_sd3 import generate_image_sd3
    from image_models.image_model_fl2 import generate_image_flux
    
    test_image = "sample_image.jpg"
    test_prompt = "A model in a futuristic outfit"

    print("Running pipeline with Stable Diffusion 3...")
    sd3_output = run_pipeline_sd3(test_image, test_prompt)
    print(f"Output from SD3 model: {sd3_output}")

    print("Running pipeline with Flux model...")
    flux_output = run_pipeline_flux(test_image, test_prompt)
    print(f"Output from Flux model: {flux_output}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LookBuilder Pipeline or Ping Listener')
    parser.add_argument('--mode', choices=['pipeline', 'ping'], 
                       default='pipeline', help='Mode to run: pipeline or ping listener')
    
    args = parser.parse_args()
    
    if args.mode == 'ping':
        run_ping_listener()
    else:
        run_pipeline()
