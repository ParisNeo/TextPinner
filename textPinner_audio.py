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
import torch
# if you don't have clip, just download it 
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
import clip 
from PIL import Image
import numpy as np

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
    text = r.recognize_google(audio_data, language='en')
    print(text)

# We are going to use only the text part of Clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Change this to put the list of your anchor texts
anchor_texts = ["raise right hand", "raise left hand", "nod", "shake hands", "look left", "look right"]
anchor_texts_tokenized = clip.tokenize(anchor_texts).to(device)

# Here is the command you have issued. Let's say you said it in french
command_text = [text] # Try other ways to say this
command_text_tokenized = clip.tokenize(command_text).to(device)

with torch.no_grad():
    # Now let's encode both text sets
    anchor_texts_embedding = model.encode_text(anchor_texts_tokenized) # Anchor texts
    command_text_embedding = model.encode_text(command_text_tokenized) # Just one text

    #Now let's measure the distances. I have tryed the mean square distance, but you may try to play around with other distances
    dist = [np.square((anchor_texts_embedding[i,:]-command_text_embedding[0,:])).mean() for i in range(len(anchor_texts))]

# Finally let's give which text is the right one
print(f"The anchor text you are searching for is {anchor_texts[np.argmin(dist)]}")

