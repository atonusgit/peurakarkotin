class Gate:
  def __init__(self):
    self.certainty = 0
    self.certainty_limit = 5
    self.pause_time = 30
    self.pause_time_limit = 30
    self.confidence_threshold = 0.5

  def is_open(self, results):
    gate_is_open = False

    print("certainty: " + str(self.certainty) + '/' + str(self.certainty_limit))
    print("pause_time: " + str(self.pause_time) + '/' + str(self.pause_time_limit))

    if self.pause_time >= self.pause_time_limit:
      for i in results:
        if "deer" in i["name"]:
          if i["confidence"] > self.confidence_threshold:
            self.certainty = self.certainty + 1

    if self.certainty > self.certainty_limit:
      self.certainty = 0
      self.pause_time = 0
      gate_is_open = True

    self.pause_time = self.pause_time + 1

    return gate_is_open
