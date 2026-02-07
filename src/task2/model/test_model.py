import pytest
import torch
import torch.nn.functional as F



from model import ICDRetrievalModel
import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional, Dict, Tuple 
import sys, os
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from helpers.image_encoder import ImageEncoder
from helpers.text_encoder import TextEncoder
from helpers.loss import contrastive_loss 
from helpers.fusion import FusionModule
from helpers.projection import ProjectionModule
# from helpers.dataset import MultimodalDataset

from transformers import AutoTokenizer, AutoModel

from task2.data.dataset import create_dataloaders



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_encoder_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')  # pretrained DINO backbone
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
text_encoder_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")   # pretrained MedCLIP text encoder

image_encoder = ImageEncoder(image_encoder_model)
text_encoder = TextEncoder(text_encoder_model, tokenizer)

# Define fusion model
img_dim = 384          # DINO CLS
text_dim = 762         # MedCLIP
projected_dim = 512        # projected fused embedding

concatenated_dim = img_dim + text_dim
fusion_module = FusionModule(method="concat") 

# Projection for fused image+text embeddings and ICD embeddings to common space
query_projection = ProjectionModule(input_dim=concatenated_dim, output_dim=projected_dim, method="linear")
icd_projection = ProjectionModule(input_dim=text_dim, output_dim=projected_dim, method="linear")

# Initialize model 
model = ICDRetrievalModel(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    fusion_module=fusion_module,
    projection_module=query_projection,
    icd_projection_module=icd_projection,
    embedding_dim=512,
    fusion_method="concat"
).to(device)



# Assuming your model file is icd_model.py and contains ICDRetrievalModel
# from icd_model import model, ICDRetrievalModel

# ---------------------------
# Mocking for fast tests
# ---------------------------

# Replace heavy encoders with simple mocks
model.encode_image = lambda x: torch.randn(1, 384)
model.encode_text = lambda x: torch.randn(1, 762)
model.encode_icd_text = lambda x: torch.randn(1, 762)

# ---------------------------
# 1. Forward pass tests
# ---------------------------
def test_forward_shapes():
    dummy_image = "/home/saad/UNIBO_AI/Sem_3/Text_Mining/Text_Mining_Project/icd-multimodal/src/task2/data/exp_data/cat_01.jpeg"
    dummy_text = "Patient has chest pain"
    dummy_icd = "I20.0 - Unstable angina"

    out = model.forward(image_path=dummy_image, query_text=dummy_text, icd_description=dummy_icd, return_all=True)
    assert 'query_embedding' in out and out['query_embedding'].shape == (1, 512)
    assert 'icd_embedding' in out and out['icd_embedding'].shape == (1, 512)
    assert 'raw_image_emb' in out and out['raw_image_emb'].shape == (1, 384)
    assert 'raw_text_emb' in out and out['raw_text_emb'].shape == (1, 762)
    assert 'fused_emb' in out

    # Image only
    out_img = model.forward(image_path=dummy_image)
    assert out_img['query_embedding'].shape == (1, 512)

    # Text only
    out_text = model.forward(query_text=dummy_text)
    assert out_text['query_embedding'].shape == (1, 512)

    # Neither -> should raise ValueError
    with pytest.raises(ValueError):
        model.forward()

# ---------------------------
# 2. Fusion tests
# ---------------------------
def test_fusion_behavior():
    img_emb = torch.randn(1, 384)
    text_emb = torch.randn(1, 762)

    # Both modalities
    fused = model.fuse_embeddings(img_emb, text_emb)
    assert fused.shape[0] == 1

    # Image only
    fused_img = model.fuse_embeddings(img_emb, None)
    assert torch.allclose(fused_img, img_emb)

    # Text only
    fused_text = model.fuse_embeddings(None, text_emb)
    assert torch.allclose(fused_text, text_emb)

    # None -> error
    with pytest.raises(ValueError):
        model.fuse_embeddings(None, None)

# ---------------------------
# 3. ICD database encoding
# ---------------------------
def test_icd_database_encoding():
    dummy_icds = ["I20.0 - Unstable angina", "I10 - Essential hypertension"]
    icd_embs = model.encode_icd_database(dummy_icds)
    assert icd_embs.shape == (2, 512)

# ---------------------------
# 4. Retrieval
# ---------------------------
def test_retrieve():
    query_emb = F.normalize(torch.randn(1, 512), dim=-1)
    icd_embs = F.normalize(torch.randn(10, 512), dim=-1)

    top_sims, top_idxs = model.retrieve(query_emb, icd_embs, top_k=3)
    assert top_sims.shape[0] == 3
    assert top_idxs.shape[0] == 3

# ---------------------------
# 5. Encode query
# ---------------------------
def test_encode_query():
    emb = model.encode_query(image_path="dummy", query_text="text")
    assert emb.shape == (1, 512)


if __name__ == "__main__":
    test_forward_shapes()
    test_fusion_behavior()
    test_icd_database_encoding()
    test_retrieve()
    test_encode_query()
    print("All tests passed!")