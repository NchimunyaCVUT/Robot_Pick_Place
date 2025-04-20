import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
from logging_setup import setup_logging
import toml

class RealSenseCamera:
    def __init__(self, logging_config, frames_dir="captured_frames"):
        """Initialize the RealSense camera with configuration"""
    
        self.log = setup_logging('REALSENSE_CAMERA', logging_config)        
        config = toml.load("config.toml")
        gemini_config = config.get("gemini", {})
        self.frames_dir = gemini_config.get("image_directory") 
        self.frame_count = 0
        self.last_capture_time = 0
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.log.info("Camera streams configured")
        
    def init_directory(self):
        """Create frame directory and clean old frames"""

        # Create directory for saving frames if it doesn't exist
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
            self.log.info(f"Created directory: {self.frames_dir}")
        
    def start(self):
        """Start streaming from the camera"""

        self.pipeline.start(self.config)
        self.last_capture_time = time.time()
        self.log.info("Camera streaming started")
        self.running = True
        
    def get_frames(self):
        """Get frames from the camera"""
        try:
            # Wait for a coherent pair of frames: depth and color
            self.log.debug("Waiting for frames...")
            self.frames = self.pipeline.wait_for_frames()
            depth_frame = self.frames.get_depth_frame()
            color_frame = self.frames.get_color_frame()
            
            # Check if both frames are valid
            if not depth_frame or not color_frame:
                return None, None, None         
            
            #Convert images to numpy arrays for input to OpenCV
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
            # Stack both images horizontally
            combined_image = np.hstack((color_image, depth_colormap))
                   
            return depth_colormap, color_image, combined_image
       
        except Exception as e:
            self.log.error(f"Error capturing frames: {e}")
            return None, None, None
        
            
    def save_frame(self, rgb_image, depth_image, frame_dir: str, frame_count=0): 
        """Clean old frames before saving new ones"""
        count = 0
        for filename in os.listdir(frame_dir):
            file_path = os.path.join(self.frames_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        self.log.info(f"Removed {count} old frames")

        #Save the depth frame and colour frame to a folder        
        self.log.info(f"Removed {count} old frames")  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_filename = f"{self.frames_dir}/frame_color_{timestamp}.jpg"
        depth_filename = f"{self.frames_dir}/frame_depth_{timestamp}.jpg"
        cv2.imwrite(rgb_filename, rgb_image)
        self.log.info(f"Captured frame {frame_count} saved as {rgb_filename}")
        cv2.imwrite(depth_filename, depth_image)
        self.log.info(f"Captured frame {frame_count} saved as {depth_filename}")
        return True       
        
    def stop(self):
        """Stop streaming and clean up resources"""
        self.log.info("Stopping camera stream")
        if self.running:
            self.pipeline.stop()
            self.running = False
        cv2.destroyAllWindows()
        self.log.info("Camera stream stopped")
