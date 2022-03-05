import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

DICT_COLORS = {
    "karolinska": {
        0: "background or unknown",
        1: "benign tissue",
        2: "cancerous tissue",
    },
    "radboud": {
        0: "background or unknown",
        1: "stroma",
        2: "healthy",
        3: "cancerous epothelium (Gleason 3)",
        4: "cancerous epothelium (Gleason 4)",
        5: "cancerous epothelium (Gleason 5)",
    },
}
