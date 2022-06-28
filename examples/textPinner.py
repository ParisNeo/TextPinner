"""TextPinner
Author : Saifeddine ALOUI
Description : Pins text to a set of anchor texts. This is useful to build a natural language command module.
Image I have a robot. For example the InMoov robot can do a bunch of actions but you some how need to ask it to do something by saying exactly what you have already specified as possible inputs.
But what if you can say the command how ever you like it, and the robot understands which one of the actions is the one you intended it to?

This code, uses Open-AI's Clip to encode both the text you say, and the list of anchor texts describing what the robot can do.
Now we simply find the nearest anchor text encoding to the encoding of the text you said.
Bingo, now you have anchored the text to waht the robot can do and you can say the command how ever you like it. The robot will understand.


You can imagine doing this for a hue lighting system or other command based app. Be creative
"""
from TextPinner import TextPinner

tp = TextPinner(["raise right hand", "raise left hand", "nod", "shake hands", "look left", "look right"], 0.5)

# you can place any text. For example if you put put your hand in my hand, the TextPinner will pin it to shake hands as it has the closest meaning
text_command = input("Input a text :")
output_text, index, similarity=tp.process(text_command)

# If index <0 then the text meaning is too far from any of the anchor texts. You can still use np.argmin(dists) to find the nearest meaning.
# or just change the maximum_distance parameter in your TextPinner when constructing TextPinner

if index>=0:
    # Finally let's give which text is the right one
    print(f"\nThe anchor text you are searching for is:\n{output_text}\n")
else:
    # Text is too far from any of the anchor texts
    print(f"\nYour text meaning is very different from the anchors meaning. Please try again\n")

for txt, sim in zip(tp.anchor_texts, similarity.tolist()):
    print(f"text : {txt}\t\t similarity \t\t {sim:0.2f}")
