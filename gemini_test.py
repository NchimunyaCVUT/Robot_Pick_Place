import json
import os
import glob
from google import genai
from google.genai import types
import time
import cv2
import numpy as np
from datetime import datetime
from logging_setup import setup_logging
import toml
from PIL import Image, ImageDraw
from apriltag_homography_1 import ApriltagHomography
import logging

class LoggingConfig:
    level = logging.INFO
    name_char_length = 12
    level_char_length = 8
    file_logging = True
    log_dir = "logs"
    max_file_size_mb = 10
    backup_count = 5

class GeminiProcessor:
    """Class for processing images with Google's Gemini API to detect objects,
    calculate centroids, and apply homography transformations."""
       
    def __init__(self, logging_config):
        """Initialize the Gemini API processor"""
        config = toml.load("config.toml")
        gemini_config = config.get("gemini", {})
        self.output_dir = gemini_config.get("Output_Directory") 
        self.results_dir = gemini_config.get("Results_Directory")
        self.image_path = gemini_config.get("image_directory") 
        self.model_name = gemini_config.get("model_name")
        self.bounding_box_system_instructions = gemini_config.get("bounding_box_system_instructions")
        self.client = genai.Client(api_key=gemini_config.get("google_api_key"))
        self.prompt = gemini_config.get("prompt")
        self.safety_settings = [types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",threshold="BLOCK_ONLY_HIGH",),]
        self.log = setup_logging('GEMINI_PROCESSOR', logging_config)
        self.init_directories()
        self.log.info("GeminiProcessor initialized with model: %s", self.model_name)
        self.colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]
        logging_config = LoggingConfig()
        self.apriltag = ApriltagHomography(logging_config)
        self.apriltag.load_tag_coordinates("world_points.json")
        
    def init_directories(self):
        """Create necessary output directories"""

        for directory in [self.output_dir, self.results_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                self.log.info(f"Created directory: {directory}")

    def parse_json(self, json_output: str):
        """Parse the JSON output from the Gemini API
        
        Args:
            json_output (str): The JSON output string from the Gemini API.
        
        Returns:
            str: The parsed JSON string.
        """
        if not json_output or not isinstance(json_output, str):
            return "[]"
        
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
               
        return json_output
    
    def  calculate_centroid(self, box_2d):
        """Calculate the centroid of a bounding box_2d
        
        Args:
            box_2d (list): List of coordinates representing the bounding box_2d. [y1, x1, y2, x2]
        
        Returns:
            tuple: The x and y coordinates of the centroid (centroid_x, centroid_y).
        """
        y1, x1, y2, x2 = box_2d
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return [cx, cy]
    
    def save_json(self, json_data, file_name=None):
        """Save JSON data to a file
        
        Args:
            json_data (dict): The JSON data to save.
            file_name (str, optional): The name of the image associated with the JSON data. Defaults to None.

        Returns:
            str: The file path where the JSON data was saved.

        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"response_{timestamp}" if file_name is None else f"{file_name}_{timestamp}"
        file_path = os.path.join(self.results_dir, f"{base_name}.json")
        
        try:
            # Parse the JSON data to ensure it's valid
            parsed_json = json.loads(json_data)

            # Write the json file
            with open(file_path, 'w') as f:
                json.dump(parsed_json, f, indent=4)
                self.log.info(f"JSON response saved to {file_path}")
            return file_path, parsed_json
        
        except json.JSONDecodeError as e:
            self.log.error(f"Invalid JSON format: {e}")
            with open(file_path, 'w') as f:
                f.write(json_data)
            self.log.warning(f"Saved raw text to {file_path}")
            return file_path, None
  
    def plot_bounding_boxes(self, image, bounding_boxes):
        """Plot bounding boxes and centroids on the image"""
        
        # Check if image is a string path or already an Image object
        if isinstance(image, str):
            # It's a file path, load the image
            img = Image.open(image)
        else:
            # It's already an Image object
            img = image
            
        # Get image dimensions
        width, height = img.size
        self.log.info(f"Image size: {width}x{height}")
        
        # Create drawing object
        draw = ImageDraw.Draw(img)
        
        # Parse JSON and process boxes
        box_2D = self.parse_json(bounding_boxes)
        try:
            boxes = json.loads(box_2D)
        except json.JSONDecodeError as e:
            self.log.error(f"Invalid JSON: {e}")
            return
        
        for i, box_2d in enumerate(boxes):
            if "box_2d" not in box_2d:
                continue  # Skip entries without box_2d
                
            color = tuple(self.colors[i % len(self.colors)])
            y1, x1, y2, x2 = box_2d["box_2d"]
            
            # Convert to absolute coordinates
            abs_y1 = int(y1/1000 * height)
            abs_x1 = int(x1/1000 * width)
            abs_y2 = int(y2/1000 * height)
            abs_x2 = int(x2/1000 * width)
            
            # Ensure correct order
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
                
            # Draw rectangle (proper format for PIL)
            draw.rectangle((abs_x1, abs_y1, abs_x2, abs_y2), outline=color, width=2)
            
            # Add label if present
            if "label" in box_2d:
                draw.text((abs_x1 + 8, abs_y1 + 6), box_2d["label"], fill=color)
                
        # Save and show image
        output_path = os.path.join(self.output_dir, f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        img.save(output_path)
        self.log.info(f"Saved image with bounding boxes to {output_path}")
        img.show()
        return img
    
    def get_depth(self, pixel_x, pixel_y, depth_image: np.ndarray, r=4):
        pixels_depth = []
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if i**2 + j**2 <= r**2:
                    depth = depth_image[pixel_y + i, pixel_x + j]
                    pixels_depth.append(depth)
        return np.mean(pixels_depth) if pixels_depth else None

    def run(self, depth_image: np.ndarray):
        """Run the Gemini API to process a single image in the directory"""

        client = self.client
        model = self.model_name 
        image_path = self.image_path

        try:
            image_files = glob.glob(f"{image_path}/*.*")
            if not image_files:
                 self.log.error(f"No files found in {image_path}")
                 return False
            
            image = image_files[0]
            self.log.info(f"Processing image: {image}")
            
            img_cv = cv2.imread(image)
            im = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            width, height = im.size
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            prompt = self.prompt 
            response = client.models.generate_content(
                model=model,
                contents=[prompt, im],
                config=types.GenerateContentConfig(
                    system_instruction=self.bounding_box_system_instructions,
                    temperature=0.5,
                    safety_settings=self.safety_settings,
                    )
                ) 
            if response.text:
                self.log.info(f"Received response from Gemini API {response.text}")
                self.plot_bounding_boxes(im, response.text)
                parsed_json_text = self.parse_json(response.text)
                data = json.loads(parsed_json_text)
                if data and isinstance(data, list):
                    self.robot_coordinates = None
                    for i, box_2d in enumerate(data):
                        if "box_2d" in box_2d:
                            self.centroid = self.calculate_centroid(box_2d["box_2d"])
                            self.log.info(f"Centroid of {box_2d['label']} is {self.centroid}")
                            self.pixel_x = int(self.centroid[0] / 1000 * width) # convert normalised coordinates to pixel coordinates
                            self.pixel_y = int(self.centroid[1] / 1000 * height)
                            self.log.info(f"Centroid in pixels is {self.pixel_x}, {self.pixel_y}")
                            self.apriltag.detect_tags(img_cv)
                            self.homography = self.apriltag.compute_homography()
                            # print(self.homography)
                            self.x_y_coordinates = self.homography @ np.array([self.pixel_x, self.pixel_y, 1]).reshape(3,1)
                            # print(f"X, Y coordinates are {x_y_coordinates}")
                            self. world_z = self.get_depth(self.pixel_x, self.pixel_y, depth_image)
                            self.world_x = round(float(self.x_y_coordinates[0,0]), 2)
                            self.world_y = round(float(self. x_y_coordinates[1,0]), 2)
                            # print(f"x coordinate is {world_x}")
                            # print(f"y coordinate is {world_y}")
                            self.robot_coordinates = [self.world_x, self.world_y, self.world_z]
                            self.log.info(f"Robot coordinates: {self.robot_coordinates}")
                            return self.robot_coordinates
                    else:
                        self.log.error("No bounding boxes found in the response.")
                    return False     
        except Exception as e:
            self.log.error(f"Error processing image: {e}")
            return False
            
