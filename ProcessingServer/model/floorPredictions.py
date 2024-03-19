class FloorPredictions:
    def __init__(self, predictions, num_labels):
        self.__num_labels = num_labels
        self.__predictions = predictions
    
    def get_predictions(self):
        return self.__predictions
    
    def get_num_labels(self):
        return self.__num_labels