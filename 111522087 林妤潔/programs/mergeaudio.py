from moviepy.editor import *
import cv2 as cv
import numpy as np
import os

def merge_advd(outputname):
    forward_path = os.path.abspath(os.path.dirname(os.getcwd()))
    video = VideoFileClip(f"{forward_path}/data/output_videos/no_audio/{outputname}.mp4")
    audio = AudioFileClip(f"{forward_path}/data/sound/sound.mp3")

    output = video.set_audio(audio)
    output.write_videofile(f"{forward_path}/data/output_videos/audio/{outputname}.mp4", temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    print('ok')
