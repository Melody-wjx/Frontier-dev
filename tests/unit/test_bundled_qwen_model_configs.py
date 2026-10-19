"""Qwen configs must remain usable without the parent SwiftMind checkout."""

import json
from pathlib import Path
import shutil

import pytest

from frontier.config.model_config import BaseModelConfig


MODEL_NAMES = ("Qwen2.5-Coder-7B", "Qwen3-Coder-30B-A3B-Instruct")
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "data" / "config" / "models"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_config_is_a_bundled_regular_file(name):
    source = CONFIG_ROOT / f"{name}.json"
    assert not source.is_symlink(), "Published configs must not reference external files"
    assert source.is_file()
    assert isinstance(json.loads(source.read_text(encoding="utf-8")), dict)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_config_loads_without_parent_checkout(name, tmp_path, monkeypatch):
    source = CONFIG_ROOT / f"{name}.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    isolated_root = tmp_path / "standalone" / "data" / "config" / "models"
    isolated_root.mkdir(parents=True)
    # Preserve symlinks so this fixture cannot accidentally hide a regression.
    shutil.copy2(source, isolated_root / source.name, follow_symlinks=False)
    monkeypatch.chdir(tmp_path / "standalone")

    model = BaseModelConfig.create_from_name(name)

    assert model.num_layers == payload["num_hidden_layers"]
    assert model.embedding_dim == payload["hidden_size"]
    assert model.num_q_heads == payload["num_attention_heads"]
    assert model.num_kv_heads == payload["num_key_value_heads"]
    expected_mlp = payload.get("moe_intermediate_size", payload["intermediate_size"])
    assert model.mlp_hidden_dim == expected_mlp
