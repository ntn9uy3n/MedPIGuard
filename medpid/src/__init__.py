"""
MedPIDefense — Deep Learning Baseline for Medical Prompt Injection Detection.

Pipeline:
    (doctor_question, patient_answer)
        -> BiomedBERT Encoder -> h_q, h_a
        -> Feature Fusion -> z = [h_q; h_a; |h_q-ha|; h_q*h_a; cos(h_q,h_a)]
        -> StandardScaler
        -> [MLP / LSTM-MLP / Autoencoder / BiLSTM]
        -> Prediction (normal / attack)
"""

__version__ = "1.0.0"
