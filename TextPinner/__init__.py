"""TextPinner
Author      : Saifeddine ALOUI
Licence     : MIT
Description : 
Pins text to a set of anchor texts. This is useful to build a natural language command module.
Image I have a robot. For example the InMoov robot can do a bunch of actions but you some how need to ask it to do something by saying exactly what you have already specified as possible inputs.
But what if you can say the command how ever you like it, and the robot understands which one of the actions is the one you intended it to?

This code, uses Google's BERT model to encode both the text you say, and the list of anchor texts describing what the robot can do.
Now we simply find the nearest anchor text encoding to the encoding of the text you said.
Bingo, now you have anchored the text to waht the robot can do and you can say the command how ever you like it. The robot will understand.


You can imagine doing this for a hue lighting system or other command based app. Be creative
"""
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EncodedAnchorsList():
    def __init__(self, anchor_texts:str, tokenizer, encoder):
        self.anchor_texts = anchor_texts
        self.tokenizer = tokenizer
        self.encoder = encoder
        if len(anchor_texts)>0:
            self.anchor_texts_tokenized = self.tokenizer(anchor_texts,add_special_tokens = True,    truncation = True, padding = "max_length", return_attention_mask = True, return_tensors='pt')#[self.tokenizer(reference_token,add_special_tokens = True,    truncation = True, padding = "max_length", return_attention_mask = True, return_tensors='pt') for reference_token in anchor_texts]
        else:
            raise Exception("Empty anchors list!")

        with torch.no_grad():
            # Now let's encode both text setssummed = torch.sum(masked_embeddings, 1)
            self.anchor_texts_embedding = torch.sum(self.encoder(**self.anchor_texts_tokenized).last_hidden_state, 1).cpu().detach() # [ for encoded_input in self.anchor_texts_tokenized] # Anchor texts


class TextPinner():
    def __init__(self, minimum_similarity_level:float=None, force_cpu=False):
        """Builds the TextPinner

        Args:
            anchor_texts (list[str]) : The list of anchor texts to pin text to
            minimum_similarity_level (float) : The minimum acceptable similarity between the text and the anchors (to avoid pinning to wrong texts when completely different command is issued)
            force_cpu(bool) : Force using CPU even when a cuda device is available
        """
        
        self.minimum_similarity_level = minimum_similarity_level
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained("bert-base-uncased")



    def buildEncodedAnchorsList(self, anchor_texts:list)->EncodedAnchorsList:
        """Sets the anchor texts to be compared with input sentences

        Args:
            anchor_texts (list): A list of strings 
        """
        return EncodedAnchorsList(anchor_texts, self.tokenizer, self.encoder)

    def setEncodedAnchorsList(self, encodedAnchorsList:EncodedAnchorsList)->None:
        self.encodedAnchorsList = encodedAnchorsList


    def process(self, command_text:str):
        """Processes text ang gives the text entended to

        Args:
            command_text (str): The command text to pin to one of the texts list

        Returns:
            str, int, list: _description_
        """
        encoded_input = self.tokenizer(command_text, return_tensors='pt')
        with torch.no_grad():
            pred = self.encoder(**encoded_input)
            command_text_embedding = torch.sum(pred.last_hidden_state, 1).cpu().detach()

        similarity = 2*cosine_similarity(command_text_embedding.numpy(), self.encodedAnchorsList.anchor_texts_embedding.numpy())[0,:]# np.array([2*cosine_similarity(command_text_embedding.numpy(), emb.numpy()) for emb in self.anchor_texts_embedding])
        max = similarity.max()
        text_index = similarity.argmax()
        if self.minimum_similarity_level is not None:
            if max<self.minimum_similarity_level:
                return "",-1, similarity
        return self.encodedAnchorsList.anchor_texts[text_index], text_index, similarity

