from view.gui import GUI
from service.connectionService import ConnectionService
from service.perceptionService import PerceptionService
from controller.connectionController import ConnectionController
from controller.perceptionController import PerceptionController


def main():
  HOST, PORT = "localhost", 9000
  windowName = "Robot Perception"
  floor_confidence_treshold = 0.5
  object_confidence_treshold = 0.7
  connection_service = ConnectionService(HOST, PORT)
  connection_controller = ConnectionController(connection_service)
  perception_service = PerceptionService(floor_confidence_treshold, object_confidence_treshold)
  perception_controller = PerceptionController(perception_service)
  GUI(windowName, perception_controller, connection_controller)  
if __name__ == "__main__":
  main()
