import torch
from model.drivePath import DrivePath
from model.floorPredictions import FloorPredictions
from data.labeling import nonInterestingLabels
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shapely.geometry import box
import copy

class PerceptionService:
    def __init__(self, floor_confidence_treshold, object_confidence_treshold) -> None:
        self.floor_confidence_treshold = floor_confidence_treshold
        self.object_confidence_treshold = object_confidence_treshold
        self.objects = []
        pass
    
    def find_objects(self, message):
        results = message.process_objects()
        self.objects = copy.deepcopy(results)
        for result in results:
            mask = result.boxes.data[:, 4] >= self.object_confidence_treshold
            result.boxes.data = result.boxes.data[mask]
        return results
    
    def find_floor(self, message):
        logits = message.process_floor()
        probabilities = torch.softmax(logits, dim=1) # Apply softmax to convert logits to probabilities
        # Get the maximum probabilities and their corresponding indices (predictions)
        max_probs, predictions = torch.max(probabilities, dim=1)
        # Filter out low confidence predictions by setting them to a background label (e.g., 0)
        # Note: Adjust the background label as needed
        background_label = 0
        predictions[max_probs < self.floor_confidence_treshold] = background_label
        predictions_np = predictions.cpu().numpy()
    
        # Exclude specific labels
        for label in nonInterestingLabels:
            predictions_np[predictions_np == label] = background_label  # Set excluded labels to background

        num_labels = logits.shape[1]  # Adjust based on your model's output
        # Now `predictions_np` only includes high-confidence predictions and excludes certain labels
        return FloorPredictions(predictions_np, num_labels)
    
    def compute_driveable_path_coords(self, img):
        return DrivePath(img)
    
    def is_path_driveable(self, img, floor):
        if (type(floor) == str):
            return False
        floor_predictions = floor.get_predictions()[0]
        height, width = img.shape[:2]
        print(f"[{width}, {height}]")
        print(f"---{floor_predictions.shape}")
        matrix_height, matrix_width = floor_predictions.shape[:2]
        drive_path = DrivePath(img).toMatrixCoords(matrix_width, matrix_height)
        return self.__check_drive_path_matrix(drive_path, floor_predictions)

    def find_path_obstructing_objects(self, img, objects):
        obstructs_path = False
        drive_path = DrivePath(img)
        for bbox in self.__process_object_boxes(self.objects):
            if bbox.intersects(drive_path.polygon):
                obstructs_path = True
        return obstructs_path

    def __process_object_boxes(self, objects):
        boxes = []
        for object in objects:
            if (object.boxes and len(object.boxes)):
                bbox = object.boxes.xyxy[0, :4].tolist()
                # Convert bounding box to a shapely polygon object
                bbox_polygon = box(*bbox)  # Unpacks the bbox list into the box function
                boxes.append(bbox_polygon)
        return boxes

    def __check_drive_path_matrix(self,drive_path, floor_matrix):
        
        pt1 = drive_path.end_left  # Top-left
        pt2 = drive_path.end_right  # Top-right
        pt3 = drive_path.start_right  # Bottom-right
        pt4 = drive_path.start_left  # Bottom-left
        x1 = pt1[0]
        x2 = pt2[0]
        x3 = pt3[0]
        x4 = pt4[0]
        floor_space = True
        """
        # Create a matrix filled with zeros
        matrix = np.zeros((drive_path.height, drive_path.width, 3), dtype=np.uint8)
        # Fill the entire matrix with green color first
        matrix[:, :] = [255, 255, 255]  # RGB for green

        for i in range(0, len(floor_matrix)):
            for j in range(0, len(floor_matrix[i])):
                if (floor_matrix[i][j] == 0):
                    matrix[i][j] = [0, 255, 255]  
                else:
                    matrix[i][j] = [0, 0, 0]  
        """
        for x in range(min(x1, x4), max(x2, x3) + 1):
            top_y = int(round(self.__interpolate_y(x, pt1, pt2)))
            bottom_y = int(round(self.__interpolate_y(x, pt4, pt3)))
            for y in range(top_y, bottom_y + 1):
                #matrix[y-1][x-1] = [255, 255, 0] 
                if (floor_matrix[y-1][x-1] == 0):
                    #matrix[y-1][x-1] = [255,255, 255]
                    floor_space = False
        #matrix = cv2.resize(matrix, (500,500), interpolation=cv2.INTER_AREA)
        # Display the matrix
        #cv2.imshow('matrix', matrix)
        return floor_space

    def __interpolate_y(self, x, pt1, pt2):
        # Linear interpolation between points pt1 and pt2
        if pt1[0] == pt2[0]:  # Avoid division by zero if vertical line
            return min(pt1[1], pt2[1])
        slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        return slope * (x - pt1[0]) + pt1[1]