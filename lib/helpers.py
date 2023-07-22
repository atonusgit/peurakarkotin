import os
import cv2
import time
import numpy
import pygame
from dotenv import load_dotenv

load_dotenv()

def save_deer_image(image):
  frame_array = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
  filename = f'image_{int(time.time())}.jpg'
  cv2.imwrite(os.getenv("PEURAHAVAINNOT_DIRECTORY") + filename, frame_array)

def play_hus_audio():
  audio_file = "audio/hus.wav"

  pygame.init()
  pygame.mixer.init()

  try:
    sound = pygame.mixer.Sound(audio_file)
    sound.play()

    while pygame.mixer.get_busy():
      continue

  except pygame.error as e:
    print(f"Error playing the sound: {e}")
