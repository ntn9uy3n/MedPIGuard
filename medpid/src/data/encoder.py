"""
TextPairEmbedder — extract BERT hidden representations for doctor-patient pairs.
"""

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import ENCODER_MODEL, MAX_LENGTH, POOLING
from src.utils import ensure_dir


class TextPairEmbedder:
    """
    Encode doctor questions and patient answers using a pre-trained BERT model.

    Args:
        model_name: HuggingFace model ID.
        device: 'cuda' or 'cpu'.
        max_length: max token length per text.
        pooling: 'cls' (CLS token) or 'mean' (mean pooling over tokens).
    """

    def __init__(
        self,
        model_name: str = ENCODER_MODEL,
        device: str = None,
        max_length: int = MAX_LENGTH,
        pooling: str = POOLING,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  Loading encoder: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into fixed-dimensional embeddings.

        Args:
            texts: list of strings
            batch_size: encoding batch size

        Returns:
            [N, D] numpy array of embeddings
        """
        all_embeddings = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state

            if self.pooling == "cls":
                emb = last_hidden[:, 0, :]
            elif self.pooling == "mean":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                emb = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            all_embeddings.append(emb.detach().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def encode_pairs(
        self,
        questions: list,
        answers: list,
        batch_size: int = 32,
    ) -> tuple:
        """
        Encode doctor questions and patient answers separately.

        Returns:
            (hq, ha) — both [N, D] numpy arrays
        """
        hq = self.encode_texts(questions, batch_size=batch_size)
        ha = self.encode_texts(answers, batch_size=batch_size)
        return hq, ha
