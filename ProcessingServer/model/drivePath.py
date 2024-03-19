
from shapely.geometry import Polygon

class DrivePath:
    def __init__(self, img):
        self.height, self.width = img.shape[:2]
        # Calculate starting points for the lines
        self.start_left = ((self.width // 2) - 120, self.height)
        self.start_right = ((self.width // 2) + 120, self.height)
        self.end_left = (self.width // 2 - 90, self.height // 2 + 160)  # Adjust the horizontal offset as needed
        self.end_right = (self.width // 2 + 90, self.height // 2 + 160)
        polygon = [self.start_left, self.end_left, self.end_right, self.start_right]
        self.polygon = Polygon(polygon)
    
    #converts the all the image coordinates to equivalent coordinates in the matrix
    def toMatrixCoords(self, matrix_width, matrix_height):
         # Calculate the scale factors for width and height
        width_scale = matrix_width / self.width
        height_scale = matrix_height / self.height
        
        # Convert starting and ending points for left line
        self.start_left = (int(self.start_left[0] * width_scale), int(self.start_left[1] * height_scale))
        self.end_left = (int(self.end_left[0] * width_scale), int(self.end_left[1] * height_scale))
        
        # Convert starting and ending points for right line
        self.start_right = (int(self.start_right[0] * width_scale), int(self.start_right[1] * height_scale))
        self.end_right = (int(self.end_right[0] * width_scale), int(self.end_right[1] * height_scale))
        self.height = matrix_height
        self.width = matrix_width
        return self
        """self.start_left = (math.floor(self.start_left[0]/self.width * matrix_width), math.floor(self.start_left[1]/self.height * matrix_height))
        self.start_right = (math.floor(self.start_right[0]/self.width * matrix_width), math.floor(self.start_right[1]/self.height * matrix_height))
        self.end_left = (math.floor(self.end_left[0]/self.width), math.floor(self.end_left[1]/self.height * matrix_height)) 
        self.end_right = (math.floor(self.end_right[0]/self.width), math.floor(self.end_right[1]/self.height * matrix_height)) 
        self.height = matrix_height
        self.width = matrix_width
        return self"""