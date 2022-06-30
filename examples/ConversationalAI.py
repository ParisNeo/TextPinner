"""TextPinner
Author : Saifeddine ALOUI
Description : A simple predefines answers chatbot using TextPinner to make it capable of understanding sentences in natural way
"""
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

# Let's build the inputs and outputs list
conversation={
    "Hi":["Hi how are you?", "Hi", "Hello there"],
    "What's your name?":[f"My name is TextPinner. I am an AI based on Open AI's CLIP to pin your commands to a set of predefined ones to help you interact with me using natural language.", "My name is TextPinner but you can call me TP."],
    "How old are you?":[f"I am {datetime.now()-datetime(2022, 7, 1)} old"]
}

# Let's create the TextPinner and the texts list
tp = TextPinner(list(conversation.keys()), 0.55)

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
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)  # ms

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
        audio_data = r.listen(source)
        print("Recognizing...")
        # convert speech to text
        try:
            text_command = r.recognize_google(audio_data, language='en')
            print(text_command)
        except:
            print("Please say an english sentence")
            continue

            

    # you can place any text. For example if you put put your hand in my hand, the TextPinner will pin it to shake hands as it has the closest meaning
    output_text, index, similarity=tp.process(text_command)

    # If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
    # or just change the maximum_distance parameter in your TextPinner when constructing TextPinner

    if index>=0:
        for key in conversation.keys():
            if output_text==key:
                rnd = np.random.randint(0,len(conversation[key]))
                output = conversation[key][rnd]
                print(output)
                say(output)
                break
        # Finally let's give which text is the right one
        if output_text=="exit":
            print(f"\nbye bye.\n")
            say("bye bye")
            exit_app=True
    else:
        # Text is too far from any of the anchor texts
        print("Sorry. I don't understand")
        say("Sorry. I don't understand")

    for txt, sim in zip(tp.anchor_texts, similarity.tolist()):
        print(f"text : {txt}\t\t similarity \t\t {sim:0.2f}")



