import time
from lib.gate import Gate
from lib.model import Model
from lib.camera import Camera
import lib.helpers as helpers

if __name__ == "__main__":
  cam = Camera()
  mod = Model()
  model = mod.get_model()
  gate = Gate()

  time.sleep(5)

  try:
    while True:
      image = cam.take_and_return_image()
      results = model.predict(source=image, save=False, save_txt=False, conf=0.5, verbose=False)

      if results[0].boxes.data.any():
        results_list = mod.get_results(model.names, results)

        if gate.is_open(results_list):
          print(time.ctime() + " - Peurahavainto")
          helpers.save_deer_image(image)
          helpers.play_hus_audio()
#          image.show()

      time.sleep(1)

  finally:
    print('end')
