from picamera2 import Picamera2, Preview

class Camera:
  def __init__(self):
    self.picam2 = Picamera2()
    self.capture_config = self.picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    self.picam2.configure(self.capture_config)
    self.picam2.start()

  def take_and_return_image(self):
    return self.picam2.capture_image("main")
