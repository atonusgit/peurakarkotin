import os
from ultralytics import YOLO
from dotenv import load_dotenv

class Model:
  def get_model(self):
    load_dotenv()
    model = YOLO('models/' + os.getenv("PEURAMODEL"))
    return model

  def get_results(self, names, results):
    result = {}
    results_list = []
    boxes = results[0].boxes
    confidence, class_ids = boxes.conf, boxes.cls.int()
    rects = boxes.xyxy.int()

    for ind in range(boxes.shape[0]):
      result['name'] = names[class_ids[ind].item()]
      result['confidence'] = confidence[ind].item()
      result['rects'] = rects[ind].tolist()
      results_list.append(result)

    return results_list
