"""PJ
Author : Saifeddine ALOUI
Description : A simple AI called PJ that uses TextPinner to interact with a user
"""
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from TextPinner import TextPinner
# gtts is needed to generate text to speech
# pygame is needed for audio mixer to output the sound
from datetime import datetime
import webbrowser


class PJ():
    def __init__(self):
        
        # Let's create the TextPinner object
        self.tp = TextPinner(0.85)

        # Now build multiple dictionaries to describe commands and the responses PJ should give for each one
        self.ai_name_detection_dict=    {
                "pj":[self.summened],
                "exit":[self.exitapp]
            } 
        # Let's build the inputs and outputs list. Outputs will be selected randomly. You can have functions as outputs so that the AI may perform some tasks
        self.main_menu_dict={
            "pj":["at your service"],
            "what time is it?":[self.show_time],
            "what date is it?":[self.show_date],
            "open a web page":[self.open_web_page],
            "how old are you?":[f"I am {(datetime.now()-datetime(2022, 7, 1)).days} days old"],
            "create a file":[self.create_file],
            "delete a file":[self.delete_file],
            "help":[self.help],    
            "exit":[self.exitapp]
        }
        # Feel free to add new interactions or even make the AI search the Web for you or interact with hardware, etc. The only limit is your imagination.
        self.say("PJ conversational tool V 1.0")
        self.say("Building vocabulary, please stand by...")
        self.ai_name_detection_embeddings = self.tp.buildEncodedAnchorsList(list(self.ai_name_detection_dict.keys()))
        self.cancel_detection_embeddings = self.tp.buildEncodedAnchorsList(["cancel"])
        self.yesno_detection_embeddings = self.tp.buildEncodedAnchorsList(["yes","no"])
        self.main_menu_embeddings = self.tp.buildEncodedAnchorsList(list(self.main_menu_dict.keys()))        

        main_menu_sentences=None
        self.active_menu=self.ai_name_detection_dict
        # We build vocabulary and interactions
        self.tp.setEncodedAnchorsList(self.ai_name_detection_embeddings)
        self.say("Ready")
        self.say("Hello. My name is PJ. You can summon me just by saying my name.")
        print("User>>>")

    def log_exception(self, ex):       
        type_, value_, traceback_ = sys.exc_info()
        msg = "{}\n{}\n{}\n{}\n".format(ex,type_, value_, '\n'.join(traceback.format_tb(traceback_)))
        print(msg)


    # A useful function to say stuff using text to speech synthesis
    def say(self, text):
        print(text)

    def listen(self, prompt):
        return input(prompt)

    def setActiveMenu(self, menu_tests, menu_embeddings):
        self.active_menu = menu_tests
        self.tp.setEncodedAnchorsList(menu_embeddings)

    def summened(self):
        responses = ["Sir","At your service","yes?"]
        self.respond(responses)
        self.setActiveMenu(self.main_menu_dict,self.main_menu_embeddings)


    def respond(self, answers):
        rnd = np.random.randint(0,len(answers))
        output = answers[rnd]
        if type(output)==str:
            self.say(output)
            self.setActiveMenu(self.ai_name_detection_dict, self.ai_name_detection_embeddings)
        else: # Sometimes we want the AI to do something instead of just answering
            output()


    def get_a_parameter(self, prompt:str)->str:
        """Gets a parameter of a command vocally. The user can cancel the operation if he says cancel, in which case a None is returned, other wize the text parameter is returnes

        Args:
            prompt (str): A prompt to show the user

        Returns:
            str: The parameter pronounced by the user or Non if the parameter is not good
        """
        cmd = self.listen(prompt)
        print("User>>>")
        # Now we only need to detect if the user wants to cancel
        self.tp.setEncodedAnchorsList(self.cancel_detection_embeddings)
        output_text, index, similarity= self.tp.process(cmd)
        if similarity[0]<0.9: # Not cancel
            return cmd
        else:
            return None


    # Main menu =================================================================
    # Few tasks to illustrate how the AI may do things on demand
    def create_file(self):
        Path("tmp").mkdir(exist_ok=True)
        fn = self.get_a_parameter("\nWhat is the file name you whish to use?\n")
        if fn is not None: # Not cancel
            data = self.get_a_parameter("\nWhat should I write in the file?\n")
            print("User>>>")

            with open(f"tmp/{fn}.txt","w") as f:
                f.write(data)
            self.say(f"File {fn} created successfully")
        else:
            self.say("Command canceled")
        self.setActiveMenu(self.ai_name_detection_dict, self.ai_name_detection_embeddings)

    def delete_file(self):
        fn = self.get_a_parameter("\nWhat is the file you whish to remove?\n")
        if fn is not None: # Not cancel
            is_sure = self.listen("Are you sure ?\n")
            print("User>>>")
            # Now we only need to detect if the user wants to cancel
            self.tp.setEncodedAnchorsList(self.yesno_detection_embeddings)
            output_text, index, similarity= self.tp.process(is_sure)
            if index==0:
                os.remove("tmp/{fn}.txt")
                self.say("File deleted successfully")
            elif index==1:
                self.say("File deletion canceled")
            else:
                self.say("Not understood, file deletion canceled")
        else:
            self.say("File deletion canceled")
        self.setActiveMenu(self.ai_name_detection_dict, self.ai_name_detection_embeddings)

    def show_time(self):
        t = datetime.now()
        self.say(f"It is {t.strftime('%I:%M %p')}")
        self.setActiveMenu(self.ai_name_detection_dict, self.ai_name_detection_embeddings)

    def show_date(self):
        t = datetime.now()
        self.say(f"It is {t.strftime('%m/%d/%Y ')}")
        self.setActiveMenu(self.ai_name_detection_dict, self.ai_name_detection_embeddings)

    def open_web_page(self):
        url = self.listen("Which web page you want me lo open ?\n")
        print("User>>>")
        self.tp.setEncodedAnchorsList(self.cancel_detection_embeddings)
        output_text, index, similarity= self.tp.process(url)
        if similarity[0]<0.9: # Not cancel
            self.say(f"Opening webpage {url}")
            webbrowser.open(url)
        else:
            self.say("Command canceled")
        self.setActiveMenu(self.ai_name_detection_dict, self.ai_name_detection_embeddings)


    def help(self):
        self.say("Here are my anchor texts. You do not need to say these commands exactly. I can anderstand your intent.")
        self.say("Known commands:") 
        for key in self.main_menu_dict.keys():
            self.say(key)


    def exitapp(self):
        self.say("bye bye")
        exit()

    def main_loop(self):
        while True:
            # convert speech to text
            text_command = self.listen("")
            # you can place any text. For example if you put put your hand in my hand, the TextPinner will pin it to shake hands as it has the closest meaning
            output_text, index, similarity=self.tp.process(text_command)

            # If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
            # or just change the maximum_distance parameter in your TextPinner when constructing TextPinner
            print("\nUser>>",end="")
            print(text_command)
            print("PJ>>",end="")


            if index>=0:
                for key in self.active_menu.keys():
                    if output_text==key:
                        self.respond(self.active_menu[key])
                        break    
            else:
                # Text is too far from any of the anchor texts
                print("\n")
            print("User>>",end="")



if __name__=="__main__":
    PJ().main_loop()