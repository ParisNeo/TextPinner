"""TextPinner
Author : Saifeddine ALOUI
Description : A simple predefines answers chatbot using TextPinner to make it capable of understanding sentences in natural way
"""
import os
from pathlib import Path
import numpy as np
from TextPinner import TextPinner

from datetime import datetime

# Let's create the TextPinner and the texts list
tp = TextPinner(0.60)

main_menu_sentences=None
# A useful function to say stuff using text to speech synthesis
def say(text):
    print(text)

# We build vocabulary and interactions
print("Building vocabilaries")
cancel_detection = tp.buildEncodedAnchorsList(["cancel"])
yesno_detection = tp.buildEncodedAnchorsList(["yes","no"])

# Few tasks to illustrate how the AI may do things on demand
def create_file():
    Path("tmp").mkdir(exist_ok=True)
    fn = input("\nWhat is the file name you whish to use?\nUser>>>")
    # Now we only need to detect if the user wants to cancel
    tp.setEncodedAnchorsList(cancel_detection)
    output_text, index, similarity= tp.process(fn)
    if similarity[0]<0.8: # Not cancel
        with open(f"tmp/{fn}.txt","w") as f:
            f.write("File created !")
        say("File created successfully")
    else:
        say("Command canceled")
    tp.setEncodedAnchorsList(main_menu_sentences)

def delete_file():
    fn = input("Are you sure ?\nUser>>>")
    # Now we only need to detect if the user wants to cancel
    tp.setEncodedAnchorsList(yesno_detection)
    output_text, index, similarity= tp.process(fn)
    if index==0:
        os.remove("tmp/file.txt")
        say("File deleted successfully")
    elif index==1:
        say("File deletion canceled")
    else:
        say("Not understood, file deletion canceled")
    tp.setEncodedAnchorsList(main_menu_sentences)

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
    "Clip":["Yes!"],
    "Hello":["Hi how are you?", "Hi", "Hello there"],
    "I am fine":["I am pleased to meet you","Happy that you are fine"],
    "What's your name?":[f"My name is TextPinner. I am an AI based on Google Bert encoder to pin your commands to a set of predefined ones to help you interact with me using natural language.", "My name is TextPinner but you can call me TP."],
    "How old are you?":[f"I am {(datetime.now()-datetime(2022, 7, 1)).days} days old"],
    "Create a file":[create_file],
    "Delete a file":[delete_file],
    "help":[help],    
    "exit":[exitapp]
}
# Feel free to add new interactions or even make the AI search the Web for you or interact with hardware, etc. The only limit is your imagination.
main_menu_sentences = tp.buildEncodedAnchorsList(list(conversation.keys()))

tp.setEncodedAnchorsList(main_menu_sentences)
print("DONE")


exit_app = False

while not exit_app:
    # read the audio data from the default microphone
    print("Listening, Say something...")
    print("User>>",end="")
    # convert speech to text
    try:
        text_command = input(">>>")
    except:
        continue

            

    # you can place any text. For example if you put put your hand in my hand, the TextPinner will pin it to shake hands as it has the closest meaning
    output_text, index, similarity=tp.process(text_command)
    print(f"nearest meaning : {output_text} ({similarity[index]*100}%)")
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



