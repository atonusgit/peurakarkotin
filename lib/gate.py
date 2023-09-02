import os
import time
import lib.helpers as helpers
from dotenv import load_dotenv

class Gate:
  def __init__(self):
    load_dotenv()
    self.certainty = 0
    self.certainty_reset = 0

  def is_open(self, results, mod, model):
    gate_is_open = False

    if results[0].boxes.data.any():
      results_list = mod.get_results(model.names, results)

      print("certainty: " + str(self.certainty) + '/' + str(os.getenv("CONFIDENCE_THRESHOLD")))

      for i in results_list:
        if "deer" in i["name"]:
          if i["confidence"] > float(os.getenv("CONFIDENCE_THRESHOLD")):
            helpers.play_audio("audio/beep2.wav", 1)
            time.sleep(0.1)
            self.certainty = self.certainty + 1
            self.certainty_reset = 0

    self.certainty_reset = self.certainty_reset + 1

    if self.certainty_reset > int(os.getenv("CERTAINTY_RESET_LIMIT")):
      self.certainty = 0
      self.certainty_reset = 0

    if self.certainty > int(os.getenv("CERTAINTY_LIMIT")):
      self.certainty = 0
      gate_is_open = True

    return gate_is_open
