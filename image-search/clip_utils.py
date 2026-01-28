import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass
class ClipBundle:
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str
    model_name: str
    pretrained: str


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai") -> ClipBundle:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return ClipBundle(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        pretrained=pretrained,
    )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def encode_image(bundle: ClipBundle, image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_input = bundle.preprocess(image).unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        embedding = bundle.model.encode_image(image_input)
    embedding = embedding.detach().cpu().numpy().astype(np.float32)
    return _l2_normalize(embedding)[0]


def encode_text(bundle: ClipBundle, text: str) -> np.ndarray:
    tokens = bundle.tokenizer([text])
    tokens = tokens.to(bundle.device)
    with torch.no_grad():
        embedding = bundle.model.encode_text(tokens)
    embedding = embedding.detach().cpu().numpy().astype(np.float32)
    return _l2_normalize(embedding)[0]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
