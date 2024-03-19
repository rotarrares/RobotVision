import cv2
import torch
import numpy as np

class GUI:
    def __init__(self, windowName, perceptionController, connectionController):
        self.perceptionController = perceptionController
        self.connectionController = connectionController
        self.color_palette = None
        self.windowName = windowName
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        self.connectionController.listen(self.drawFrame)
    
    def drawFrame(self, frame_data):
        image = self.processFrame(frame_data)
        cv2.imshow(self.windowName, image)

    def processFrame(self, frame_data):
        img = frame_data.toCvImage()
        results = self.perceptionController.find_objects(frame_data)
        floor = self.perceptionController.find_floor(frame_data)
        path_obstructed = self.perceptionController.find_obstructed_path(img, floor, results)
        objected_image = self.display_object_results(img, results)
        segmented_image = self.display_segmentation_results(objected_image, floor)
        final_image = self.add_visual_tracks(segmented_image, path_obstructed)
        return final_image

    def display_segmentation_results(self, img, floor):
        # Extract the tensor from the dictionary
        # Step 1: Convert tensor to numpy array
        if (type(floor)) == str:
            print(f'Warning :{floor}')
            return img
        # Now `predictions_np` only includes high-confidence predictions and excludes certain labels
        # Generate color map
        if self.color_palette is None:
            self.color_palette = np.random.randint(0, 255, (floor.get_num_labels(), 3), dtype=np.uint8)

        color_map = self.color_palette[floor.get_predictions()]  # Assuming predictions is your argmax result
        color_map_image = color_map.squeeze()  # Remove the batch dimension if present
        height, width = img.shape[:2]
        color_map_resized = cv2.resize(color_map_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Convert original image to RGB (OpenCV uses BGR by default)
        original_image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        # Blend the original image with the color map
        alpha = 0.6  # Adjust alpha to control the transparency
        overlayed_image = cv2.addWeighted(original_image_rgb, alpha, color_map_resized, 1 - alpha, 0)

        # Display the overlayed image
        return overlayed_image
    
    
    def display_object_results(self, img, object_results):
        
        if img is not None:
            
            for result in object_results:
                img = result.plot(img=img, boxes=True, probs=True, labels=True, conf=True, masks=True)
        return img
    
    def add_visual_tracks(self, img, path_obstructed):
        if type(img) == str: 
            print(f'Warning :{img}')
            return img
        # Calculate starting points for the lines
        driveable_path = self.perceptionController.get_driveable_path_coords(img)
        line_color =  (20,70, 255)
        if path_obstructed:
            line_color = (255, 70, 20)
        # Draw the left line
        print(path_obstructed)
        cv2.line(img, driveable_path.start_left, driveable_path.end_left, line_color, thickness=5)  # Blue color in BGR

        # Draw the right line
        cv2.line(img, driveable_path.start_right, driveable_path.end_right, line_color, thickness=5)  # Blue color in BGR
        return img
