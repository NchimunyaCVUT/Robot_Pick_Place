import cv2
import numpy as np
import logging
from apriltag_homography_1 import ApriltagHomography
import cvzone

class LoggingConfig:
    level = logging.INFO
    name_char_length = 12
    level_char_length = 8
    file_logging = True
    log_dir = "logs"
    max_file_size_mb = 10
    backup_count = 5

logging_config = LoggingConfig()
apriltag = ApriltagHomography(logging_config) 
image = cv2.imread("./camera_frames/frame_5_20250412_210208.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
world_points = apriltag.load_tag_coordinates("world_points.json")
detector = apriltag.detect_tags(image)
homography = apriltag.compute_homography()
detected_tags = apriltag.draw_tags(image.copy())
print(homography)

cv2.imshow("Original", image)
cv2.imshow("Detected AprilTags", detected_tags)
cv2.waitKey(0)
cv2.destroyAllWindows()
