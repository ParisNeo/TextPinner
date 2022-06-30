"""TextPinner
Author : Saifeddine ALOUI
Description : A simple predefines answers chatbot using TextPinner to make it capable of understanding sentences in natural way
"""
import os
from pathlib import Path
import sys
import time

import numpy as np
from TextPinner import TextPinner
from io import BytesIO
# gtts is needed to generate text to speech
from gtts import gTTS
# pygame is needed for audio mixer to output the sound
import pygame
from datetime import datetime
pygame.mixer.init()

# You need to install speach recognition engine
# pip install SpeechRecognition
import speech_recognition as sr
# You also need pyaudio for using your microphone
# You can find it here https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio select your version download it the ninstall it using pip install -e <the wheel file>


# A useful function to say stuff using text to speech synthesis
def say(text):
    # Language in which you want to convert
    language = 'en'
    
    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang=language, slow=False)
    tts.write_to_fp(mp3_fp)
    pygame.mixer.music.load(mp3_fp, "mp3")
    pygame.mixer.music.play()  # works fine !
    print(text)
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)  # ms

# Few tasks to illustrate how the AI may do things on demand
def create_file():
    Path("tmp").mkdir(exist_ok=True)
    with open("tmp/file.txt","w") as f:
        f.write("File created !")
    say("File created successfully")
def delete_file():
    os.remove("tmp/file.txt")
    say("File deleted successfully")    
def help():
    say("Here are my anchor texts. You do not need to say these commands exactly. I can anderstand your intent.")
    print("Simple Conversational AI")
    print("Known commands:") 
    for key in conversation.keys():
        print(key)  
def exitapp():
    say("bye bye")
    exit()    


# Let's build the inputs and outputs list. Outputs will be selected randomly. You can have functions as outputs so that the AI may perform some tasks
conversation={
    "Hello":["Hi how are you?", "Hi", "Hello there"],
    "I am fine":["I am pleased to meet you","Happy that you are fine"],
    "What's your name?":[f"My name is TextPinner. I am an AI based on Open AI's CLIP to pin your commands to a set of predefined ones to help you interact with me using natural language.", "My name is TextPinner but you can call me TP."],
    "How old are you?":[f"I am {(datetime.now()-datetime(2022, 7, 1)).days} old"],
    "Create a file":[create_file],
    "Delete a file":[delete_file],
    "help":[help],
    
    "exit":[exitapp]
}
# Feel free to add new interactions or even make the AI search the Web for you or interact with hardware, etc. The only limit is your imagination.

# Let's create the TextPinner and the texts list
tp = TextPinner(list(conversation.keys()), 0.80)


exit_app = False
while not exit_app:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # wait for a second to let the recognizer
        # adjust the energy threshold based on
        # the surrounding noise level
        r.adjust_for_ambient_noise(source, duration=0.2)    
        # read the audio data from the default microphone
        print("Listening, Say something...")
        print("User>>",end="")
        audio_data = r.listen(source,timeout=20)
        # convert speech to text
        try:
            text_command = r.recognize_google(audio_data, language='en')
            print(text_command)
        except:
            print("\nAI>>",end="")
            say("Please say an english sentence")
            continue

            

    # you can place any text. For example if you put put your hand in my hand, the TextPinner will pin it to shake hands as it has the closest meaning
    output_text, index, similarity=tp.process(text_command)

    # If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
    # or just change the maximum_distance parameter in your TextPinner when constructing TextPinner
    print("AI>>",end="")

    if index>=0:
        for key in conversation.keys():
            if output_text==key:
                rnd = np.random.randint(0,len(conversation[key]))
                output = conversation[key][rnd]
                if type(output)==str:
                    say(output)
                else: # Sometimes we want the AI to do something instead of just answering
                    output()
                break
    else:
        # Text is too far from any of the anchor texts
        print("Sorry. I don't understand")
        say("Sorry. I don't understand")



