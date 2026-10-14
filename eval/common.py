"""Small helpers shared by the GPU job and the report."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def local_path(uri: str | Path) -> Path:
    """A local copy of ``uri``. ``gs://`` objects are copied with gcloud (on the runner VM,
    ``rt-runner`` can read scratch); anything else is taken as a local path."""
    uri = str(uri)
    if not uri.startswith("gs://"):
        return Path(uri)
    if shutil.which("gcloud") is None:
        raise RuntimeError(f"{uri}: gcloud isn't installed here; pass a local path")
    dest = Path(tempfile.mkdtemp(prefix="snapcall-eval-")) / uri.rsplit("/", 1)[-1]
    subprocess.run(["gcloud", "storage", "cp", uri, str(dest)], check=True)
    return dest


def model_label(model_path: str | Path) -> str:
    """``/opt/models/Qwen/Qwen3-4B-Instruct-2507`` -> ``qwen3-4b-instruct-2507``."""
    return Path(str(model_path).rstrip("/")).name.lower()


def load_model(
    model_path: str | Path, dtype: str = "bfloat16", attn: str = "sdpa", quant: str | None = None
):
    """(model in eval mode, tokenizer). On GPU the weights go straight to the device, so a
    4B model in float32 (16 GB) never has to fit in the VM's RAM first.

    ``quant="8bit"``: bitsandbytes LLM.int8 weights (Decision 11: Phi-4 14B is ~28 GB in bf16,
    ~15 GB in 8-bit); everything that isn't quantised stays in ``dtype``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {"dtype": DTYPES[dtype], "attn_implementation": attn}
    if quant == "8bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant:
        raise ValueError(f"unknown quantisation {quant!r}")
    if torch.cuda.is_available():
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model.eval(), tokenizer


def free() -> None:
    """Return freed GPU memory to the device. Callers must drop their own references to the
    model first (``del model``): a helper can't do that for them."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_json(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, default=str) + "\n")


def device_info() -> dict:
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        info["allow_tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return info
