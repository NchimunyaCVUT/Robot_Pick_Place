import numpy as np
import cv2
import json
from logging_setup import setup_logging

class CoordinateTransformer:
    """Transforms image centroids to real-world coordinates using a homography matrix"""
    
    def __init__(self, logging_config):
        """Initialize with a homography matrix"""
        self.homography_matrix = None
        self.log = setup_logging('HOMOGRAPHY', logging_config)
    
    def transform_point(self, centroid, homography_matrix):
        """Convert a centroid point to real-world coordinates"""
        point = np.array([[[int(centroid[0]), int(centroid[1])]]], dtype=np.int32)
        transformed = cv2.perspectiveTransform(point, np.linalg.inv(homography_matrix))
        return [int(transformed[0][0][0]), int(transformed[0][0][1])]
    
    def transform_json(self, json_file_path):
        """Transform the centroid in the json file and return the results"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        results = {}
        for obj in data:
            if "centroid" in obj:
                label = obj.get("label", "unknown")
                results[label] = {
                    "image_centroid": obj["centroid"],
                    "world_coordinates": self.transform_point(obj["centroid"])
                }        
        return results