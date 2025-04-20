import cv2
from apriltag_homography_1 import ApriltagHomography
from stream_class import RealSenseCamera
from image_world import CoordinateTransformer
from gemini_test import  GeminiProcessor
from logging_setup import setup_logging
import logging
import time

class LoggingConfig:
    level = logging.INFO
    name_char_length = 12
    level_char_length = 8
    file_logging = True
    log_dir = "logs"
    max_file_size_mb = 10
    backup_count = 5

if __name__ == '__main__':
    logging_config = LoggingConfig()
    camera = RealSenseCamera(logging_config)
    apriltag = ApriltagHomography(logging_config)
    api = GeminiProcessor(logging_config)
    transformer = CoordinateTransformer(logging_config)
    # apriltag.load_tag_coordinates("world_points.json")
    camera.init_directory()  
    camera.start()     
    api.run() 
    try:
        while True:
            depth_image, color_image, combined_image = camera.get_frames() 
            markers = apriltag.detect_tags(color_image)
            image_with_markers = apriltag.draw_tags(color_image.copy())
            # homography = apriltag.compute_homography()
            cv2.imshow("Combined Image", combined_image)
            cv2.imshow("Detected AprilTags", image_with_markers) 
            # api.run(depth_image.copy())           
            if markers:
                print(f"Detected {len(markers)} AprilTags:")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Exit on 'q' or ESC
                camera.stop()
                break
                
    except KeyboardInterrupt:
            print("Program interrupted by user")
    except Exception as e:
            print(f"Error: {e}")
