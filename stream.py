import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

frames_dir = "captured_frames"
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create directory for saving frames if it doesn't exist
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

#Delete old frames in folder before saving new ones
for filename in os.listdir(frames_dir):
    file_path = os.path.join(frames_dir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path) 

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Variables for frame capture timing
last_capture_time = time.time()
frame_count = 0

# Setup logging
def setup_logging():
    """Configure logging with console and file handlers"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create logger
    logger = logging.getLogger('realsense_camera')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler with rotation at 10MB, keeping 5 backup files
    file_handler = RotatingFileHandler('logs/camera_stream.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

#********************** Function to get frames from the camera*********************#
def get_frames():
    try:
        # Wait for a coherent pair of frames: depth and color
        logger.info("Waiting for frames...")
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infrared_frame = frames.get_infrared_frame()
        logger.info("Frames captured successfully")
        return depth_frame, color_frame, infrared_frame
   
    except Exception as e:
        logger.error(f"Error capturing frames: {e}")
        logger.info("No frames captured")        
        return None, None, None
        
#********************** Function to capture and save a frame *********************#
def save_frame(image, frame_count): 
    # Save the image with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{frames_dir}/frame_{frame_count}_{timestamp}.png"
    cv2.imwrite(filename, image)
    logger.info(f"Captured frame {frame_count} saved as {filename}")

#********************** while loop to continuously stream camera ****************#
while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays for input to OpenCV
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap ))

    
    # Capture one frame per second
    current_time = time.time()
    if current_time - last_capture_time >= 1.0:  # 1 second interval
        save_frame(color_image, frame_count)
        frame_count += 1
        last_capture_time = current_time

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    # cv2.namedWindow('Infrared', cv2.WINDOW_AUTOSIZE)
    
    # Check for key press to exit
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key & 0xFF == 27:  # Exit on 'q' or ESC
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
