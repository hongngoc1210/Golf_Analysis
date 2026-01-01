import yaml
import torch

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(runtime_cfg: dict) -> str:
    device_cfg = runtime_cfg.get("device", "auto")
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg