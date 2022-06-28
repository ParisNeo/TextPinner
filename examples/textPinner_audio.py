"""TextPinner
Author : Saifeddine ALOUI
Description : Uses audio voice to pin text to a set of anchor texts. This is useful to build a natural language command module.
Image I have a robot. For example the InMoov robot can do a bunch of actions but you some how need to ask it to do something by saying exactly what you have already specified as possible inputs.
But what if you can say the command how ever you like it, and the robot understands which one of the actions is the one you intended it to?

This code, uses Open-AI's Clip to encode both the text you say, and the list of anchor texts describing what the robot can do.
Now we simply find the nearest anchor text encoding to the encoding of the text you said.
Bingo, now you have anchored the text to waht the robot can do and you can say the command how ever you like it. The robot will understand.


You can imagine doing this for a hue lighting system or other command based app. Be creative
"""
import sys
import time
from TextPinner import TextPinner
from io import BytesIO
# gtts is needed to generate text to speech
from gtts import gTTS
# pygame is needed for audio mixer to output the sound
import pygame
pygame.mixer.init()

# You need to install speach recognition engine
# pip install SpeechRecognition
import speech_recognition as sr
# You also need pyaudio for using your microphone
# You can find it here https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio select your version download it the ninstall it using pip install -e <the wheel file>

# Let's create the TextPinner and the texts list
# For example a set of actions to be performed by a robot
# tp = TextPinner(["raise right hand", "raise left hand", "nod", "shake hands", "look left", "look right"], 0.8)
# A commands to control lighting in a room
tp = TextPinner(["Turn on the light", "Turn off the light", "Increase the lighting", "Decrease the lighting"], 0.55)

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


r = sr.Recognizer()
with sr.Microphone() as source:
    # wait for a second to let the recognizer
    # adjust the energy threshold based on
    # the surrounding noise level
    r.adjust_for_ambient_noise(source, duration=0.2)    
    # read the audio data from the default microphone
    say("Please issue a command")
    print("Listening...")
    audio_data = r.listen(source)
    print("Recognizing...")
    # convert speech to text
    try:
        text_command = r.recognize_google(audio_data, language='en')
        print(text_command)
    except:
        print("Please say an english sentence")
        sys.exit(1)

        

# you can place any text. For example if you put put your hand in my hand, the TextPinner will pin it to shake hands as it has the closest meaning
output_text, index, similarity=tp.process(text_command)

# If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
# or just change the maximum_distance parameter in your TextPinner when constructing TextPinner

if index>=0:
    # Finally let's give which text is the right one
    say(f"The nearest anchor text to your query is {output_text}")
    print(f"\nThe anchor text you are searching for is:\n{output_text}\n")
else:
    # Text is too far from any of the anchor texts
    say(f"Your text meaning is very different from the anchors meaning. Please try again")    
    print(f"\nYour text meaning is very different from the anchors meaning. Please try again\n")

for txt, sim in zip(tp.anchor_texts, similarity.tolist()):
    print(f"text : {txt}\t\t similarity \t\t {sim:0.2f}")



