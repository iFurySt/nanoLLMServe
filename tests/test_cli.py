from types import SimpleNamespace

import nanollmserve.cli.generate as generate_cli
from nanollmserve.cli.generate import build_parser


def test_cli_parser_accepts_minimal_generate_args():
    args = build_parser().parse_args(
        [
            "--model",
            "/models/Qwen3-1.7B",
            "--prompt",
            "hello",
        ]
    )

    assert args.model == "/models/Qwen3-1.7B"
    assert args.prompt == "hello"
    assert args.max_new_tokens == 32
    assert args.temperature == 0.0
    assert args.device == "auto"
    assert args.dtype == "auto"


def test_cli_main_prints_text_and_stats(monkeypatch, capsys):
    def fake_load_model_and_tokenizer(model, *, device, dtype, local_files_only):
        assert model == "/models/Qwen3-1.7B"
        assert device == "cpu"
        assert dtype == "float32"
        assert local_files_only is True
        return SimpleNamespace(model=object(), tokenizer=object(), device=device, dtype=dtype)

    def fake_generate_one(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        assert prompt == "hello"
        assert max_new_tokens == 3
        assert temperature == 0.7
        assert seed == 11
        return SimpleNamespace(
            text="world",
            prompt_tokens=1,
            generated_tokens=3,
            elapsed_seconds=0.5,
            tokens_per_second=6.0,
            ttft_seconds=0.2,
            tpot_seconds=0.15,
        )

    monkeypatch.setattr(generate_cli, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(generate_cli, "generate_one", fake_generate_one)

    exit_code = generate_cli.main(
        [
            "--model",
            "/models/Qwen3-1.7B",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "3",
            "--temperature",
            "0.7",
            "--seed",
            "11",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--local-files-only",
            "--show-stats",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "world\n"
    assert "prompt_tokens=1 generated_tokens=3" in captured.err
    assert "tokens_per_second=6.00 ttft_seconds=0.200 tpot_seconds=0.150" in captured.err
    assert "device=cpu dtype=float32" in captured.err
