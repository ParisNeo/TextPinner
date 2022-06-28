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
from TextPinner import TextPinner

# You need to install speach recognition engine
# pip install SpeechRecognition
import speech_recognition as sr
# You also need pyaudio for using your microphone
# You can find it here https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio select your version download it the ninstall it using pip install -e <the wheel file>

r = sr.Recognizer()
with sr.Microphone() as source:
    # wait for a second to let the recognizer
    # adjust the energy threshold based on
    # the surrounding noise level
    r.adjust_for_ambient_noise(source, duration=0.2)    
    # read the audio data from the default microphone
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
# Now that we have the text, let's pin it
tp = TextPinner(["raise right hand", "raise left hand", "nod", "shake hands", "look left", "look right", "jump"])
output_text, index, probs, dists=tp.process(text_command)
# Finally let's give which text is the right one
if index>=0:
    # Finally let's give which text is the right one
    print(f"The anchor text you are searching for is {output_text}")
else:
    print(f"Your text meaning is very far from the anchors meaning. Please try again or change the minimal accepted distance value")


for txt, prob, dist in zip(tp.anchor_texts, probs, dists):
    print(f"text : {txt}\t\t prob \t\t {prob:0.2f}, dist \t\t {dist:0.2f}")
