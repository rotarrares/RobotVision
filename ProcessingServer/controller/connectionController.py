

class ConnectionController:
    def __init__(self, service):
        self.service = service

    def listen(self, drawFrame):
        try:
            self.service.listen(drawFrame)
        except KeyboardInterrupt:
            print("Server is shutting down.")
        finally:
            self.service.close_socket()
    