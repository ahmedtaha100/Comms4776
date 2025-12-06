import torch
import timm
import open_clip
import platform
from pathlib import Path

def main():
    info = {}
    info["python_version"] = platform.python_version()
    info["torch_version"] = torch.__version__
    info["timm_version"] = timm.__version__
    info["open_clip_version"] = open_clip.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
        info["cuda_device_count"] = device_count
        info["cuda_devices"] = devices
    else:
        info["cuda_device_count"] = 0
        info["cuda_devices"] = []
    info["project_root"] = str(Path(__file__).resolve().parent.parent)
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
