# text_encoder.py
import torch
import torch.nn.functional as F

class TextEncoder:
    def __init__(self, model, tokenizer):
        """
        model: pretrained MedCLIP / text encoder
        """
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self, text):
        """
        text: string
        returns: normalized embedding
        """

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        return cls_embedding
