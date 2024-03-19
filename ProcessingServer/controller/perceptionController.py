
class PerceptionController:
    def __init__(self, service):
        self.service = service
        self.floor_data = None
        self.objects_data = None
        pass

    
    def find_objects(self, frame_data):
        try:
            objects_data = self.service.find_objects(frame_data)
            return objects_data
        except:
            return f"Couldn't find objects"
    
    def find_floor(self, frame_data):
        try:
            return self.service.find_floor(frame_data)
        except:
            return "Couldn't detect floor"
        
        
    def find_obstructed_path(self, img, logits, objects):
        self.service.is_path_driveable(img, logits)
        self.service.find_path_obstructing_objects(img, objects)
        return self.service.is_path_driveable(img, logits)
    
    def get_driveable_path_coords(self, img):
        return self.service.compute_driveable_path_coords(img)