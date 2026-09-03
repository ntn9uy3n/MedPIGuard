"""
Configuration module — all hyperparameters, paths, and constants.
"""

import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # medpid/
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models_output"
RESULTS_FILE = PROJECT_ROOT / "baseline_results.json"

# Dataset
TRAIN_CSV = DATA_DIR / "trainset.csv"
TEST_CSV = DATA_DIR / "testset.csv"

# ============================================================
# Encoder
# ============================================================
ENCODER_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 256
POOLING = "cls"  # "cls" or "mean"
EMBED_BATCH_SIZE = 32

# ============================================================
# Training
# ============================================================
SEED = 42
DEVICE = "cuda"  # overridden at runtime

# Supervised split
SUP_TRAIN_RATIO = 0.8
SUP_VAL_RATIO = 0.2

# Autoencoder (unsupervised) — train on normal only
AE_TRAIN_NORMAL_RATIO = 0.8

# ============================================================
# MLP Classifier
# ============================================================
MLP_HIDDEN_DIMS = [1024, 512, 256]
MLP_DROPOUT = 0.3
MLP_EPOCHS = 50
MLP_LR = 1e-3
MLP_BATCH_SIZE = 32
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 10

# ============================================================
# LSTM-MLP Classifier
# ============================================================
LSTM_NUM_CHUNKS = 4
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_MLP_HIDDEN = 256
LSTM_MLP_DROPOUT = 0.3
LSTM_EPOCHS = 50
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 32
LSTM_WEIGHT_DECAY = 1e-4
LSTM_PATIENCE = 10

# ============================================================
# BiLSTM Classifier (sequence-level)
# ============================================================
BILSTM_HIDDEN = 512
BILSTM_LAYERS = 2
BILSTM_DROPOUT = 0.3
BILSTM_EPOCHS = 50
BILSTM_LR = 1e-3
BILSTM_BATCH_SIZE = 32
BILSTM_WEIGHT_DECAY = 1e-4
BILSTM_PATIENCE = 10

# ============================================================
# Autoencoder (Anomaly Detection)
# ============================================================
AE_LATENT_DIM = 128
AE_DROPOUT = 0.15
AE_EPOCHS = 100
AE_LR = 1e-3
AE_BATCH_SIZE = 32
AE_WEIGHT_DECAY = 1e-4

# ============================================================
# Attention-MLP Classifier (Optimized)
# ============================================================
ATTN_MLP_NUM_CHUNKS = 4
ATTN_MLP_D_MODEL = 768
ATTN_MLP_NHEAD = 8
ATTN_MLP_NUM_LAYERS = 1
ATTN_MLP_DIM_FF = 512
ATTN_MLP_DROPOUT = 0.3
ATTN_MLP_STOCHASTIC_DEPTH = 0.1
ATTN_MLP_HIDDEN = [512, 256]
ATTN_MLP_DROPOUT_HEAD = 0.3
ATTN_MLP_EPOCHS = 80
ATTN_MLP_LR = 3e-4
ATTN_MLP_BATCH_SIZE = 32
ATTN_MLP_WEIGHT_DECAY = 1e-3
ATTN_MLP_PATIENCE = 20
ATTN_MLP_POS_WEIGHT = 3.0
ATTN_MLP_LABEL_SMOOTHING = 0.05
ATTN_MLP_WARMUP = 5

# ============================================================
# Transformer Encoder Classifier
# ============================================================
TRANSFORMER_NUM_CHUNKS = 4
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_DIM_FF = 1024
TRANSFORMER_DROPOUT = 0.2
TRANSFORMER_MLP_HIDDEN = 256
TRANSFORMER_MLP_DROPOUT = 0.3
TRANSFORMER_EPOCHS = 50
TRANSFORMER_LR = 5e-4
TRANSFORMER_BATCH_SIZE = 32
TRANSFORMER_WEIGHT_DECAY = 1e-4
TRANSFORMER_PATIENCE = 15

# ============================================================
# TabNet Classifier
# ============================================================
TABNET_N_STEPS = 4
TABNET_DIMS = [1024, 512]
TABNET_SPARSE_SPARSITY = 0.01
TABNET_DROPOUT = 0.2
TABNET_EPOCHS = 80
TABNET_LR = 5e-4
TABNET_BATCH_SIZE = 64
TABNET_WEIGHT_DECAY = 1e-4
TABNET_PATIENCE = 15

# ============================================================
# Feature Fusion
# ============================================================
FEATURE_FORMULA = "[hq; ha; |hq-ha|; hq*ha; cosine(hq,ha)]"
# BERT-768 -> 4*768 + 1 = 3073
