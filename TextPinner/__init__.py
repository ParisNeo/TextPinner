"""TextPinner
Author      : Saifeddine ALOUI
Licence     : MIT
Description : 
Pins text to a set of anchor texts. This is useful to build a natural language command module.
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
import numpy as np


class TextPinner():
    def __init__(self, anchor_texts:list, minimum_similarity_level:float=None):
        """Builds the TextPinner

        Args:
            anchor_texts (list[str]) : The list of anchor texts to pin text to
            minimum_similarity_level (float) : The minimum acceptable similarity between the text and the anchors (to avoid pinning to wrong texts when completely different command is issued)
        """
        self.anchor_texts = anchor_texts
        self.minimum_similarity_level = minimum_similarity_level
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, _ = clip.load("ViT-B/32", device=self.device)
        self.anchor_texts_tokenized = clip.tokenize(self.anchor_texts).to(self.device)
        with torch.no_grad():
            # Now let's encode both text sets
            self.anchor_texts_embedding = self.clip.encode_text(self.anchor_texts_tokenized) # Anchor texts
            self.anchor_texts_embedding /= self.anchor_texts_embedding.norm(dim=-1, keepdim=True)

    def process(self, command_text:str):
        """Processes text ang gives the text entended to

        Args:
            command_text (str): The command text to pin to one of the texts list

        Returns:
            str, int, list: _description_
        """
        command_text_tokenized = clip.tokenize([command_text]).to(self.device)
        with torch.no_grad():
            command_text_embedding = self.clip.encode_text(command_text_tokenized) # Just one text

        command_text_embedding /= command_text_embedding.norm(dim=-1, keepdim=True)
        similarity = (100.0 * command_text_embedding @ self.anchor_texts_embedding.T).softmax(dim=-1).detach().numpy()[0,:]
        max = similarity.max()
        if self.minimum_similarity_level is not None:
            if max<self.minimum_similarity_level:
                return "",-1, similarity
        text_index = similarity.argmax()
        return self.anchor_texts[text_index], text_index, similarity

