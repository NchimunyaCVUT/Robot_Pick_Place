import json
import cv2
import numpy as np
from logging_setup import setup_logging


class ApriltagHomography:
    """
    Class for finding April Tags in image and calculating a homography matrix
    which transforms coordinates in pixels to coordinates defined by detected April Tags.
    """

    def __init__(self, logging_config):
        """
        ApriltagHomography object constructor.
        """
        # Setup logging
        # Must happen before any logging function call
        self.log = setup_logging('HOMOGRAPHY', logging_config)

        self.world_points = None

        self.image_points = {}  # Will now store corner coordinates
        self.world_points_detect = []
        self.image_points_detect = []
        self.homography = None
        self.tag_corner_list = None
        self.tag_id_list = None

        # Create aruco detector for selected tags
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        self.log.info(f'Initialized Apriltag detector')

    def load_tag_coordinates(self, file_path: str):
        """
        Loads conveyor world corner points from a json file.
        The json should map tag IDs to a list of 4 corner coordinates (e.g., [[x1, y1, z1], ...]).
        """
        with open(file_path, 'r') as file:
            self.world_points = json.load(file)

    def detect_tags(self, rgb_image: np.ndarray):
        """
        Detects april tags in the input image and stores their corners.
        """
        assert isinstance(rgb_image, np.ndarray)

        grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Detect markers in frame
        (corners, ids, rejected) = self.aruco_detector.detectMarkers(grayscale_image)

        # If nothing was detected, return
        if len(corners) == 0 or ids is None:
            return

        self.tag_corner_list = corners
        self.tag_id_list = ids.flatten()
        self.image_points = {}  # Clear previous centroid points

        for i, tag_corners in enumerate(self.tag_corner_list):
            tag_id = str(int(self.tag_id_list[i]))
            # Reshape to get individual corner coordinates
            corners_reshaped = tag_corners.reshape((4, 2)).astype(np.float32)
            self.image_points[tag_id] = corners_reshaped.tolist()

    def compute_homography(self) -> np.ndarray:
        """
        Computes homography matrix using image and conveyor world corner points.
        """
        image_points_list = []
        world_points_list = []

        if self.tag_corner_list is not None and self.tag_id_list is not None:
            for i, tag_corners in enumerate(self.tag_corner_list):
                tag_id = str(int(self.tag_id_list[i]))
                if tag_id in self.world_points:
                    world_corners = np.array(self.world_points[tag_id], dtype=np.float32)[:, :2] # Take only x and y for homography
                    image_corners = tag_corners.reshape((4, 2)).astype(np.float32)

                    # Ensure we have 4 corresponding points
                    if len(image_corners) == 4 and len(world_corners) == 4:
                        image_points_list.extend(image_corners)
                        world_points_list.extend(world_corners)

        # Only update homography matrix if enough points were detected (at least 4 correspondences)
        enough_points_detected = len(image_points_list) >= 16  # 4 points per tag, 4 tags minimum

        if enough_points_detected:
            self.homography, _ = cv2.findHomography(
                np.array(image_points_list), np.array(world_points_list), cv2.RANSAC, 5.0
            )
            self.log.info(f'Homography matrix computed with {len(image_points_list)} correspondences')
        else:
            self.log.warning(f'Less than 16 AprilTag corner correspondences found, homography matrix was not computed')
            self.homography = None

        return self.homography

    def compute_base_depth(self, depth_image: np.ndarray) -> int | float | None:
        """
        Computes the average base depth of all detected tags using their corner projections.
        Requires the homography to be computed first.
        """
        if self.homography is None:
            self.log.warning("Homography matrix not computed, cannot compute base depth using corners.")
            return None

        all_tag_depths = []
        if self.tag_corner_list is not None and self.tag_id_list is not None:
            for i, tag_corners in enumerate(self.tag_corner_list):
                tag_id = str(int(self.tag_id_list[i]))
                if tag_id in self.world_points:
                    world_corners_3d = np.array(self.world_points[tag_id], dtype=np.float32)
                    # Project the world corners to image coordinates using the homography
                    projected_corners = cv2.perspectiveTransform(world_corners_3d[:, :2].reshape(1, -1, 2), self.homography).reshape(4, 2).astype(np.int32)

                    # Extract depth values within the bounding box of the projected corners
                    min_x = np.min(projected_corners[:, 0])
                    max_x = np.max(projected_corners[:, 0])
                    min_y = np.min(projected_corners[:, 1])
                    max_y = np.max(projected_corners[:, 1])

                    # Ensure the bounding box is within the image boundaries
                    h, w = depth_image.shape[:2]
                    min_x = max(0, min_x)
                    max_x = min(w, max_x)
                    min_y = max(0, min_y)
                    max_y = min(h, max_y)

                    if min_x < max_x and min_y < max_y:
                        tag_depth_values = depth_image[min_y:max_y, min_x:max_x].flatten()
                        if tag_depth_values.size > 0:
                            all_tag_depths.append(np.nanmean(tag_depth_values))

        if all_tag_depths:
            return np.nanmean(all_tag_depths)
        else:
            return None

    def draw_tags(self, image_frame: np.ndarray) -> np.ndarray:
        """
        Draws detected april tags into image frame.

        Args:
            image_frame (np.ndarray): Image where apriltags are to be drawn.

        Returns:
            np.ndarray: Image with drawn april tags.
        """
        assert isinstance(image_frame, np.ndarray)

        if self.tag_corner_list is None or self.tag_id_list is None:
            return image_frame

        cv2.polylines(image_frame, np.int0(self.tag_corner_list), True, (0, 255, 0), 2)

        for i, tag_id in enumerate(self.tag_id_list):
            tag_id_str = str(int(tag_id))
            if tag_id_str in self.image_points:
                # Use the first corner for text positioning (you might want to adjust this)
                text_x = int(self.image_points[tag_id_str][0][0]) + 30
                text_y = int(self.image_points[tag_id_str][0][1])
                text = tag_id_str
                cv2.putText(
                    image_frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return image_frame