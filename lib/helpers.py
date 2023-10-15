import os
import cv2
import time
import numpy
import pygame
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = f.read().strip()
        return f"{int(temp) // 1000}C"
    except FileNotFoundError:
        return "N/A"

def save_image(image, file_timestamp, directory=os.getenv("PEURAHAVAINNOT_DIRECTORY"), add_date = False):
  frame_array = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
  filename = f'peura_{file_timestamp}.jpg'

  # Add a date to the image
  if add_date:
    date_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " CPU=" + get_cpu_temp()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_array, date_text, (10 + 2, 50 + 1), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_array, date_text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # Create directory if it doesn't exist
  if not os.path.exists(directory):
    os.makedirs(directory)

  cv2.imwrite(os.path.join(directory, filename), frame_array)

def save_plotted_image(results, file_timestamp, directory=os.getenv("PEURAHAVAINNOT_DIRECTORY")):
  res_plotted = results[0].plot()
  filename = f'peura_{file_timestamp}_plotted.jpg'
  cv2.imwrite(directory + filename, res_plotted)

def save_cropped_plot_image(results, file_timestamp, directory=os.getenv("PEURAHAVAINNOT_DIRECTORY")):
  res_plotted = results[0].plot()
  filename = f'peura_{file_timestamp}_cropped_plot.jpg'
  results[0].save_crop(directory, filename)

def play_audio(audio_file, volume=0.5):
  pygame.mixer.init()

  try:
    sound = pygame.mixer.Sound(audio_file)
    sound.set_volume(volume)
    sound.play()

    while pygame.mixer.get_busy():
      continue

  except pygame.error as e:
    print(f"Error playing the sound: {e}")

def light_toggle(light_id, on_off="off"):
  pistorasiat_root = os.getenv('PISTORASIAT_ROOT_DIRECTORY')
  pistorasiat_user = os.getenv('PISTORASIAT_USERNAME')
  pistorasiat_address = os.getenv('PISTORASIAT_ADDRESS')

  os.system("ssh " + pistorasiat_user + "@" + str(pistorasiat_address) + " 'python3 " + pistorasiat_root + "/remote_control.py " + light_id + " " + on_off + "' > /dev/null 2>&1")
