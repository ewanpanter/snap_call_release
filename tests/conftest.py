import os

import pytest

REAL_WEIGHTS_MODEL = os.environ.get("SNAPCALL_REAL_WEIGHTS_MODEL", "Qwen/Qwen3-0.6B")


def pytest_addoption(parser):
    parser.addoption(
        "--real-weights",
        action="store_true",
        default=False,
        help="run @pytest.mark.real_weights tests (downloads Qwen3-0.6B; slow on a small machine)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--real-weights"):
        return
    skip = pytest.mark.skip(reason="real-weights test: CI only, run with --real-weights")
    for item in items:
        if "real_weights" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_qwen3():
    """(model, tokenizer): 2-layer random-weights Qwen3, eager attention. Don't mutate it."""
    from tests.fixtures.tiny_qwen3 import build_tiny_qwen3

    return build_tiny_qwen3()


@pytest.fixture(scope="session")
def real_qwen3():
    """(model, tokenizer) for Qwen3-0.6B, float32 on CPU, eager attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REAL_WEIGHTS_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        REAL_WEIGHTS_MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return model, tokenizer
