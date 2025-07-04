import sys
import types
from contextlib import contextmanager


def _stub_torch():
    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(Dataset=object, DataLoader=object)
    )
    torch.nn = types.SimpleNamespace(
        Module=object,
        Linear=object,
        functional=types.SimpleNamespace(pad=lambda *a, **k: None),
        Softmax=lambda *a, **k: lambda x: x,
    )
    torch.Tensor = object
    torch.device = type("device", (), {})
    torch.load = lambda *a, **k: {}
    torch.topk = lambda *a, **k: ([], [])

    @contextmanager
    def no_grad():
        yield
    torch.no_grad = no_grad
    sys.modules['torch'] = torch
    sys.modules['torch.utils'] = torch.utils
    sys.modules['torch.utils.data'] = torch.utils.data


def _stub_torchvision():
    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.SimpleNamespace(
        resnet18=lambda: types.SimpleNamespace(fc=types.SimpleNamespace(in_features=0))
    )
    torchvision.transforms = types.SimpleNamespace(
        Compose=lambda *a, **k: lambda x: x,
        Lambda=lambda f: f,
        Resize=lambda *a, **k: lambda x: x,
    )
    sys.modules['torchvision'] = torchvision


def _stub_torchaudio():
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.load = lambda *a, **k: (None, 22050)
    torchaudio.functional = types.SimpleNamespace(resample=lambda x, orig_sr, sr: x)
    torchaudio.transforms = types.SimpleNamespace(
        MelSpectrogram=lambda *a, **k: lambda x: x,
        AmplitudeToDB=lambda *a, **k: lambda x: x,
    )
    sys.modules['torchaudio'] = torchaudio


def _stub_pyaudioop():
    pyaudioop = types.ModuleType("pyaudioop")

    def _return_zero(*args, **kwargs):
        return 0

    for name in [
        "max",
        "minmax",
        "avg",
        "avgpp",
        "rms",
        "bias",
        "ulaw2lin",
        "lin2ulaw",
        "lin2lin",
        "add",
        "mul",
        "ratecv",
        "cross",
    ]:
        setattr(pyaudioop, name, _return_zero)

    sys.modules["pyaudioop"] = pyaudioop


def pytest_configure(config):
    try:
        import torch  # noqa: F401
    except Exception:
        _stub_torch()

    try:
        import torchvision  # noqa: F401
    except Exception:
        _stub_torchvision()

    try:
        import torchaudio  # noqa: F401
    except Exception:
        _stub_torchaudio()

    try:
        import pyaudioop  # noqa: F401
    except Exception:
        _stub_pyaudioop()
