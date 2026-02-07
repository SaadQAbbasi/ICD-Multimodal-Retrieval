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
        if isinstance(text, str):                           # accept single string or list of strings
            text = [text]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        device = next(self.model.parameters()).device       # move to model device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # outputs.last_hidden_state has shape [batch_size, seq_len, hidden_size]
        cls_embedding = F.normalize(cls_embedding, dim=-1)  # shape: (batch_size, hidden_size)

        if len(text) == 1:
            return cls_embedding[0] # return 1D tensor for single text
        
        return cls_embedding # return [batch_size, hidden_size] for multiple texts

