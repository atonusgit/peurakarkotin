import os
import cv2
import time
import asyncio
from lib.gate import Gate
from lib.model import Model
from lib.camera import Camera
import lib.helpers as helpers
from dotenv import load_dotenv

load_dotenv()

def reactions():
  helpers.light_toggle("I", "on")
  helpers.play_audio("audio/hus.wav", 1)
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

  time.sleep(5)

  try:
    while True:
      if pause_time >= pause_time_limit:
        image = cam.take_and_return_image()
        results = model.predict(source=image, save=False, save_txt=False, conf=float(os.getenv("CONFIDENCE_THRESHOLD")), verbose=False)

        if results[0].boxes.data.any():
          results_list = mod.get_results(model.names, results)

          if gate.is_open(results_list):
            print(time.ctime() + " - Peurahavainto")
            file_timestamp = time.ctime().replace(" ", "_").replace(":", "-")

            helpers.save_plotted_image(results, file_timestamp)
            helpers.save_image(image, file_timestamp)

            cam.stop_camera()
            asyncio.run(react(file_timestamp))
            cam.start_still_camera()
            pause_time = 0
#           image.show()

      else:
        helpers.play_audio("audio/hum.wav", 0.5)

      pause_time = pause_time + 1
      time.sleep(1)

  finally:
    print('end')
