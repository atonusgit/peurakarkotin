import os
import cv2
import time
import asyncio
from lib.gate import Gate
from lib.model import Model
from lib.camera import Camera
from datetime import datetime
import lib.helpers as helpers
from dotenv import load_dotenv

load_dotenv()

def reactions():
  helpers.light_toggle("I", "on")
  helpers.play_audio(os.getenv("AUDIO_HUS_FILE"), float(os.getenv("AUDIO_HUS_VOLUME")))
  helpers.light_toggle("I", "off")

async def async_record_video(file_timestamp):
  loop = asyncio.get_event_loop()
  await loop.run_in_executor(None, cam.capture_and_save_video, file_timestamp, 30)

async def async_reactions():
  time.sleep(1)
  loop = asyncio.get_event_loop()
  await loop.run_in_executor(None, reactions)

async def react(file_timestamp):
  await asyncio.gather(async_record_video(file_timestamp), async_reactions())

if __name__ == "__main__":
  cam = Camera()
  mod = Model()
  model = mod.get_model()
  gate = Gate()

  pause_time = 15
  pause_time_limit = pause_time

  timelapse_limit = 15
  time.sleep(5)

  try:
    while True:
      if pause_time >= pause_time_limit:
        image = cam.take_and_return_image()
        results = model.predict(source=image, save=False, save_txt=False, conf=float(os.getenv("CONFIDENCE_THRESHOLD_SENSITIVE")), verbose=False)
        file_timestamp = time.ctime().replace(" ", "_").replace(":", "-")

        # image 1 - timelapse
        if timelapse_limit > 15:
          helpers.save_image(image, file_timestamp, os.getenv("PEURAHAVAINNOT_TIMELAPSE_DIRECTORY") + datetime.now().strftime("%y%m%d") + "/", True)
          timelapse_limit = 0

        # image 2 - sensitive detection
        if gate.is_open(results, mod, model, os.getenv("CONFIDENCE_THRESHOLD_SENSITIVE"), True):
          print("sensitive")
          helpers.save_image(image, file_timestamp, os.getenv("PEURAHAVAINNOT_SENSITIVE_DETECTION_DIRECTORY") + "deer/")
          helpers.save_plotted_image(results, file_timestamp, os.getenv("PEURAHAVAINNOT_SENSITIVE_DETECTION_DIRECTORY") + "deer/")
          helpers.save_cropped_plot_image(results, file_timestamp, os.getenv("PEURAHAVAINNOT_SENSITIVE_DETECTION_DIRECTORY"))

        # image 3 - confident reaction
        if gate.is_open(results, mod, model, os.getenv("CONFIDENCE_THRESHOLD_REACTION")):
          print(time.ctime() + " - Peurahavainto")

          helpers.save_plotted_image(results, file_timestamp)
          helpers.save_image(image, file_timestamp)

          cam.stop_camera()
          asyncio.run(react(file_timestamp))
          cam.start_still_camera()
          pause_time = 0
#          image.show()

      else:
        helpers.play_audio(os.getenv("AUDIO_PAUSE_FILE"), float(os.getenv("AUDIO_PAUSE_VOLUME")))

      timelapse_limit = timelapse_limit + 1
      pause_time = pause_time + 1
      time.sleep(1)

  finally:
    print('end')
