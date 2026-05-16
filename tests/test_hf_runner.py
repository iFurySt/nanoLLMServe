from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import nanollmserve.model.hf_runner as hf_runner


def test_resolve_device_prefers_explicit_value():
    assert hf_runner.resolve_device("cpu") == "cpu"


def test_resolve_dtype_accepts_aliases():
    assert hf_runner.resolve_dtype("fp32") is torch.float32
    assert hf_runner.resolve_dtype("bf16") is torch.bfloat16


def test_resolve_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported dtype"):
        hf_runner.resolve_dtype("int8")


def test_load_model_and_tokenizer_disables_optional_transformers_deps(monkeypatch):
    calls = {"tokenizer": [], "model": []}

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_path, *, local_files_only):
            calls["tokenizer"].append((model_path, local_files_only))
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model"].append((model_path, kwargs))
            return FakeModel()

        def to(self, device):
            self.device = device

        def eval(self):
            self.did_eval = True

    fake_import_utils = SimpleNamespace(
        _scipy_available=True,
        _sklearn_available=True,
        _torchvision_available=True,
    )
    fake_transformers_utils = SimpleNamespace(import_utils=fake_import_utils)
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeModel,
        AutoTokenizer=FakeTokenizer,
    )

    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setitem(__import__("sys").modules, "transformers.utils", fake_transformers_utils)

    loaded = hf_runner.load_model_and_tokenizer(
        "/models/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        local_files_only=True,
    )

    assert calls["tokenizer"] == [("/models/Qwen3-1.7B", True)]
    assert calls["model"] == [
        (
            "/models/Qwen3-1.7B",
            {"dtype": torch.float32, "local_files_only": True},
        )
    ]
    assert fake_import_utils._scipy_available is False
    assert fake_import_utils._sklearn_available is False
    assert fake_import_utils._torchvision_available is False
    assert loaded.device == "cpu"
    assert loaded.dtype == "float32"


def test_load_model_and_tokenizer_falls_back_to_torch_dtype(monkeypatch):
    calls = []

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_path, *, local_files_only):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls.append(kwargs)
            if "dtype" in kwargs:
                raise TypeError("unexpected keyword argument 'dtype'")
            return FakeModel()

        def to(self, device):
            self.device = device

        def eval(self):
            self.did_eval = True

    monkeypatch.setenv("NANOLLMSERVE_DISABLE_OPTIONAL_TRANSFORMERS_DEPS", "0")
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeModel, AutoTokenizer=FakeTokenizer),
    )

    hf_runner.load_model_and_tokenizer(
        "/models/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        local_files_only=True,
    )

    assert calls == [
        {"dtype": torch.float32, "local_files_only": True},
        {"torch_dtype": torch.float32, "local_files_only": True},
    ]
