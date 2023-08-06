import os
import time
from dotenv import load_dotenv
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder

class Camera:
  def __init__(self):
    self.picam2 = Picamera2()
    self.capture_config = self.picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    self.picam2.configure(self.capture_config)
    self.picam2.start()
    load_dotenv()

  def take_and_return_image(self):
    return self.picam2.capture_image("main")

  def capture_and_save_video(self, file_timestamp, video_length):
    self.picam2.configure(self.picam2.create_video_configuration())
    encoder = H264Encoder(bitrate=10000000)
    filename = f'peura_{file_timestamp}.h264'

    self.picam2.start_recording(encoder, os.getenv("PEURAHAVAINNOT_DIRECTORY") + filename)
    time.sleep(video_length)
    self.picam2.stop_recording()

  def stop_camera(self):
    self.picam2.stop()

  def start_still_camera(self):
    self.picam2.configure(self.capture_config)
    self.picam2.start()
