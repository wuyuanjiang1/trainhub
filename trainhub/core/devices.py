"""跨平台的计算设备探测与解析。

``auto`` 的选择优先级：CUDA（NVIDIA 显卡）→ MPS（Apple Silicon）→ CPU。
在 Windows / Linux / macOS 上行为一致；显式选择的设备若本机不可用，
会自动回退到本机最优设备，避免训练中途才报错。
"""

from __future__ import annotations

# 计算设备参数的候选项（auto 交给 resolve_device 按平台解析）
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

DEVICE_HELP = (
    "auto 按平台自动选择：优先 CUDA（NVIDIA 显卡），其次 MPS（Apple Silicon），最后 CPU"
)


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


def detect_device() -> str:
    """返回本机当前可用的最优设备：cuda / mps / cpu。"""
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def resolve_device(choice: str) -> str:
    """把设备选项解析成 torch / nnUNet 能接受的设备名。"""
    return detect_device() if choice == "auto" else str(choice).strip().lower()


def validate_device(choice: str) -> tuple[str, str | None]:
    """校验所选设备在本机是否可用，不可用时回退到本机最优设备。

    返回 (实际使用的设备, 警告信息)；警告信息为 None 表示按原选择执行。
    """
    device = resolve_device(choice)
    if device == "cuda" and not cuda_available():
        best = detect_device()
        return best, f"所选设备 cuda 在本机不可用，已回退为 {best}"
    if device == "mps" and not mps_available():
        best = detect_device()
        return best, f"所选设备 mps 在本机不可用，已回退为 {best}"
    return device, None
