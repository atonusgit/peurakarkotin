import os
import cv2
import time
import numpy
import pygame
from dotenv import load_dotenv

load_dotenv()

def save_image(image, file_timestamp):
  frame_array = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
  filename = f'peura_{file_timestamp}.jpg'
  cv2.imwrite(os.getenv("PEURAHAVAINNOT_DIRECTORY") + filename, frame_array)

def save_plotted_image(results, file_timestamp):
  res_plotted = results[0].plot()
  filename = f'peura_{file_timestamp}_plotted.jpg'
  cv2.imwrite(os.getenv("PEURAHAVAINNOT_DIRECTORY") + filename, res_plotted)

def play_audio(audio_file, volume=0.5):
  pygame.init()
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

  os.system("ssh " + pistorasiat_user + "@" + str(pistorasiat_address) + " 'python3 " + pistorasiat_root + "/remote_control.py " + light_id + " " + on_off + "'")