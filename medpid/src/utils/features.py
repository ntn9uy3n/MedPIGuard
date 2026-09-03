"""
Pairwise feature fusion for doctor-patient dialogue embeddings.
"""

import numpy as np


def cosine_feature(hq: np.ndarray, ha: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between hq and ha.

    Args:
        hq: [N, D] — doctor question embeddings
        ha: [N, D] — patient answer embeddings

    Returns:
        [N, 1] — cosine similarity scores
    """
    numerator = np.sum(hq * ha, axis=1, keepdims=True)
    denominator = (
        np.linalg.norm(hq, axis=1, keepdims=True)
        * np.linalg.norm(ha, axis=1, keepdims=True)
        + 1e-8
    )
    return numerator / denominator


def build_pair_features(hq: np.ndarray, ha: np.ndarray) -> np.ndarray:
    """
    Feature fusion from doctor question and patient answer embeddings.

    z = [hq; ha; |hq - ha|; hq * ha; cos(hq, ha)]

    Args:
        hq: [N, D] — doctor question embeddings
        ha: [N, D] — patient answer embeddings

    Returns:
        [N, 4*D + 1] — fused feature vector

    Rationale:
        - hq, ha: raw semantic representation of each turn
        - |hq - ha|: absolute divergence (captures semantic mismatch)
        - hq * ha: element-wise interaction (shared semantic space)
        - cos(hq, ha): similarity scalar (incongruence signal)

    Prompt injection answers are semantically incongruent with medical
    doctor questions, so these interaction features amplify the signal.
    """
    diff = np.abs(hq - ha)
    prod = hq * ha
    cos = cosine_feature(hq, ha)

    z = np.concatenate([hq, ha, diff, prod, cos], axis=1)
    return z.astype(np.float32)
