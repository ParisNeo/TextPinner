# TextPinner
A tool to pin text to a specific set of texts. Useful to build a tool that takes any text input then infer which one of the anchor texts is the best one.

# Description
TextPinner pins text to a set of anchor texts. This is useful to build a natural language command module.
ImagINE I have a robot. For example the InMoov robot can do a bunch of actions but you some how need to ask it to do something by saying exactly what you have already specified as possible inputs.
But what if you can say the command how ever you like it, and the robot understands which one of the actions is the one you intended it to?

This code, uses Open-AI's Clip to encode both the text you say, and the list of anchor texts describing what the robot can do.
Now we simply find the nearest anchor text encoding to the encoding of the text you said.
Bingo, now you have anchored the text to waht the robot can do and you can say the command how ever you like it. The robot will understand.

Actually this works even when you tell the commands in another language (tryed it in French).

You can imagine doing this for a hue lighting system or other command based app. Be creative

# Installation

To install TextPinner, you can use:
```
pip install TextPinner
```
UNfortunately Clip is not a default pip package and it is mandatory for this tool to function. So you will need to download it directly from github :
```
pip install git+https://github.com/openai/CLIP.git
```

It is advised to install cudatoolkit if you have a cuda enabled GPU.
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
```


# USE
First import TextPinner class
```Python
from TextPinner import TextPinner
```

Create an instance of TextPinner. There are one mandatory parameter which is the list of anchor text, and an optional parameter which is the minimum similarity between the word and the anchors. This allows the AI to detect if the text the user is entering is too far from any of the anchors. By default the value is None (don't check for minimal distance). A value of 0.5 has proven to be a good distance for the tests we have done but this can be changed depending on the anchors you are using. Feel free to use another value :
```Python
tp = TextPinner(["raise right hand", "raise left hand", "nod", "shake hands", "look left", "look right"], minimum_similarity_level=0.5)
```

Now you are ready to pin some text using process method that returns multiple useful outputs.

```Python
text_command = input("Input a text :")
output_text, index, similarity=tp.process(text_command)
```
- The index tells you which text of your anchors list is most likely to have the same meaning as the text_command. If it is -1, this means that the meaning of the text is too far from any of the anchors. If maximum_distance is None then there is no maximum distance test and the AI will return the anchor with nearest meaning.
- output_text is literally the anchor text that has the nearest meaning to the one of text_command.
- similarity is a numpy array containing the similarity of this text with each of the anchor texts. Useful to get an idea about the certainty of the algorithm about its decision.

# Examples
In the examples section we provide two examples:
1 - A simple text pinner example where the user is prompted to enter a text and then we tell him which anchortext is the nearest in meaning to the text he typed.
2 - A voice command example that uses voice to inpuit the text then gives the most likely output