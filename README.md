# TextPinner
A tool to pin text to a specific set of texts. Useful to build a tool that takes any text input then infer which one of the anchor texts is the best one.

# Description
TextPinner pins text to a set of anchor texts. This is useful to build a natural language command module.
Image I have a robot. For example the InMoov robot can do a bunch of actions but you some how need to ask it to do something by saying exactly what you have already specified as possible inputs.
But what if you can say the command how ever you like it, and the robot understands which one of the actions is the one you intended it to?

This code, uses Open-AI's Clip to encode both the text you say, and the list of anchor texts describing what the robot can do.
Now we simply find the nearest anchor text encoding to the encoding of the text you said.
Bingo, now you have anchored the text to waht the robot can do and you can say the command how ever you like it. The robot will understand.

Actually this works even when you tell the commands in another language (tryed it in French).

You can imagine doing this for a hue lighting system or other command based app. Be creative

# Requirements

You need to install pytorch and clip from open-ai repository
'''bash
$> conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$> pip install ftfy regex tqdm
$> pip install git+https://github.com/openai/CLIP.git
'''

