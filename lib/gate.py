import lib.helpers as helpers
from dotenv import load_dotenv

class Gate:
  def __init__(self):
    load_dotenv()
    self.certainty = 0

  def is_open(self, results):
    gate_is_open = False
    print("certainty: " + str(self.certainty) + '/' + str(os.getenv("CONFIDENCE_THRESHOLD")))

    for i in results:
      if "deer" in i["name"]:
        if i["confidence"] > float(os.getenv("CONFIDENCE_THRESHOLD")):
          helpers.play_audio("audio/beep.wav", 0.25)
          self.certainty = self.certainty + 1

    if self.certainty > os.getenv("CERTAINTY_LIMIT"):
      self.certainty = 0
      gate_is_open = True

    return gate_is_open
