"""Tests de los pesos persistentes de modelos (`mohh.llamea.models`)."""

import json
import os
import sys

import pytest

from mohh.llamea import models


def _fake_model(path, *, weights=True, tokenizer=True, index=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump({"model_type": "qwen2"}, handle)
    if tokenizer:
        with open(os.path.join(path, "tokenizer.json"), "w") as handle:
            handle.write("{}")
    if weights:
        with open(os.path.join(path, "model.safetensors"), "wb") as handle:
            handle.write(b"\x00" * 16)
    if index is not None:
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as handle:
            json.dump({"weight_map": {f"w{i}": f"shard-{i}.safetensors"
                                      for i in range(index)}}, handle)
    return path


def test_models_root_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path / "explicit"))
    assert models.models_root() == os.path.abspath(str(tmp_path / "explicit"))


def test_models_root_uses_lustre(tmp_path, monkeypatch):
    monkeypatch.delenv("MOHH_MODELS_DIR", raising=False)
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("LUSTRE", str(tmp_path))
    monkeypatch.setenv("USER", "tester")
    assert models.models_root() == os.path.join(str(tmp_path), "tester", "mohh-models")


def test_safe_name_and_model_dir(tmp_path):
    assert models.safe_name("Qwen/Qwen2.5-Coder-7B") == "Qwen--Qwen2.5-Coder-7B"
    assert models.model_dir("a/b", root=str(tmp_path)) == os.path.join(str(tmp_path), "a--b")


def test_verify_model_ok_and_incomplete(tmp_path):
    good = _fake_model(str(tmp_path / "good"))
    result = models.verify_model(good)
    assert result["ok"] and result["weights"] == 1 and result["total_bytes"] > 0

    bad = _fake_model(str(tmp_path / "bad"), weights=False)
    result = models.verify_model(bad)
    assert not result["ok"]
    assert any("pesos" in problem for problem in result["problems"])


def test_verify_model_detects_missing_shard(tmp_path):
    path = _fake_model(str(tmp_path / "sharded"), index=3)
    result = models.verify_model(path)
    assert not result["ok"]
    assert any("shard" in problem for problem in result["problems"])


def test_download_model_reuses_verified_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path))
    path = _fake_model(models.model_dir("org/model", root=str(tmp_path)))
    called = {"n": 0}

    def fake_snapshot(*args, **kwargs):  # pragma: no cover - no debe llamarse
        called["n"] += 1

    monkeypatch.setitem(sys.modules, "huggingface_hub", type(
        "m", (), {"snapshot_download": staticmethod(fake_snapshot)})())
    resolved = models.download_model("org/model")
    assert resolved == path
    assert called["n"] == 0  # ya estaba verificado: no descarga
    assert models.read_manifest(path)["ok"] is True


def test_download_model_invokes_snapshot_download(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path))
    target = models.model_dir("org/new", root=str(tmp_path))

    def fake_snapshot(model_id, revision=None, local_dir=None, cache_dir=None,
                      max_workers=None):
        _fake_model(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", type(
        "m", (), {"snapshot_download": staticmethod(fake_snapshot)})())
    resolved = models.download_model("org/new", revision="main")
    assert resolved == target
    assert models.read_manifest(target)["revision"] == "main"


def test_download_model_refuses_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        models.download_model("org/absent", allow_download=False)


def test_list_models_reports_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path))
    _fake_model(models.model_dir("org/one", root=str(tmp_path)))
    found = models.list_models()
    assert any(entry["ok"] and entry["model_id"] for entry in found)


def test_configure_hf_env_points_to_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path / "models"))
    # El resto de la suite puede haber fijado estas variables; aqui se aisla.
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    root = models.configure_hf_env()
    assert os.environ["HF_HOME"] == root
    assert os.environ["HUGGINGFACE_HUB_CACHE"].startswith(root)


def test_check_free_space_refuses_when_disk_is_full(tmp_path):
    import shutil

    free = shutil.disk_usage(tmp_path).free
    models.check_free_space(str(tmp_path), 1)  # hay espacio de sobra
    with pytest.raises(RuntimeError, match="Espacio insuficiente"):
        models.check_free_space(str(tmp_path), free + 1)


def test_download_model_checks_remote_size_before_download(tmp_path, monkeypatch):
    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path))
    called = {"snapshot": 0}

    class FakeHfApi:
        def model_info(self, *args, **kwargs):
            class Info:
                siblings = [type("S", (), {"size": 10 ** 18})()]
            return Info()

    def fake_snapshot(**_kwargs):  # pragma: no cover - no debe llamarse
        called["snapshot"] += 1

    monkeypatch.setitem(sys.modules, "huggingface_hub", type(
        "m", (), {"HfApi": FakeHfApi,
                  "snapshot_download": staticmethod(fake_snapshot)})())
    with pytest.raises(RuntimeError, match="Espacio insuficiente"):
        models.download_model("org/huge")
    assert called["snapshot"] == 0


def test_cli_root_and_list(tmp_path, monkeypatch, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mohh_models_cli",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "mohh_models.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("MOHH_MODELS_DIR", str(tmp_path))
    _fake_model(models.model_dir("org/one", root=str(tmp_path)))
    module.main(["root"])
    out = capsys.readouterr().out
    assert str(tmp_path) in out
    module.main(["list"])
    assert "org/one" in capsys.readouterr().out
