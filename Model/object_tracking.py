from pathlib import Path
from sahi import AutoDetectionModel
from boxmot import DeepOCSORT

def initialize_tracker(model_weights, device='cpu'):
    return DeepOCSORT(
        model_weights=Path(model_weights),
        device=device,
        fp16=False,
    )
