import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.data import generate_train_data
from scripts.data import download_and_split
from scripts.data import shard_jsonl
from scripts.data import tokenize_official_rollouts
from scripts.data import validate_regenerated_conversations
from scripts.data.verify_jsonl_shard import verify_shard


def test_invalid_empty_user_is_removed_before_official_split():
    valid = {
        "id": 0,
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
        ],
    }
    empty_user = {
        "id": 1,
        "conversations": [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": "answer"},
        ],
    }
    empty_source_assistant = {
        "id": 2,
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": ""},
        ],
    }

    assert download_and_split.is_valid_source_conversation(valid)
    assert not download_and_split.is_valid_source_conversation(empty_user)
    assert download_and_split.is_valid_source_conversation(empty_source_assistant)
    assert download_and_split.normalize_conversations(empty_source_assistant) == {
        "id": 2,
        "conversations": [{"role": "user", "content": "question"}],
    }
    intermediate_empty_user = {
        "id": 3,
        "conversations": [
            {"from": "human", "value": "first"},
            {"from": "gpt", "value": "old answer"},
            {"from": "human", "value": ""},
            {"from": "human", "value": "next meaningful turn"},
        ],
    }
    assert download_and_split.is_valid_source_conversation(intermediate_empty_user)
    assert [
        message["content"]
        for message in download_and_split.normalize_conversations(
            intermediate_empty_user
        )["conversations"]
    ] == ["first", "old answer", "next meaningful turn"]
    recoverable_leading_placeholder = {
        "id": 4,
        "conversations": [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": "irrelevant source answer"},
            {"from": "human", "value": "first usable user turn"},
        ],
    }
    assert download_and_split.normalize_conversations(
        recoverable_leading_placeholder
    )["conversations"] == [
        {"role": "user", "content": "first usable user turn"}
    ]


def test_official_generator_regenerates_every_user_turn(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            message = SimpleNamespace(content=f"new-answer-{len(calls)}")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(generate_train_data, "OpenAI", FakeClient)
    args = SimpleNamespace(
        model="target",
        max_tokens=4096,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=None,
        repetition_penalty=None,
        enable_thinking=False,
        disable_thinking=True,
        is_reasoning_model=False,
        is_gpt_oss=False,
    )
    sample = {
        "conversations": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "old-a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "old-a2"},
        ]
    }

    result = generate_train_data.call_sglang(args, "worker:30000", sample)

    assert [message["content"] for message in result["conversations"]] == [
        "u1",
        "new-answer-1",
        "u2",
        "new-answer-2",
    ]
    assert calls[0]["messages"] == [{"role": "user", "content": "u1"}]
    assert [message["content"] for message in calls[1]["messages"]] == [
        "u1",
        "new-answer-1",
        "u2",
    ]
    assert all(call["max_tokens"] == 4096 for call in calls)


def test_exact_id_resume_processes_holes_instead_of_a_line_prefix(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": source_id,
                    "conversations": [{"role": "user", "content": str(source_id)}],
                }
            )
            for source_id in (10, 11, 12)
        )
        + "\n",
        encoding="utf-8",
    )
    output.write_text(
        json.dumps(
            {
                "id": 11,
                "status": "success",
                "conversations": [
                    {"role": "user", "content": "11"},
                    {"role": "assistant", "content": "done"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with output.open("ab") as handle:
        handle.write(b'{"id": 12, "status": "success"')
    generated_ids = []
    monkeypatch.setattr(generate_train_data, "validate_servers", lambda _args: ["worker"])

    def fake_call(_args, _server, sample, max_tokens=None):
        assert max_tokens is None
        generated_ids.append(sample["id"])
        sample["conversations"].append(
            {"role": "assistant", "content": f"a-{sample['id']}"}
        )
        sample["status"] = "success"
        return sample

    monkeypatch.setattr(generate_train_data, "call_sglang", fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_train_data.py",
            "--model",
            "target",
            "--server-address",
            "worker",
            "--input-file-path",
            str(source),
            "--output-file-path",
            str(output),
            "--concurrency",
            "2",
            "--resume",
            "--resume-by-id",
        ],
    )

    generate_train_data.main()

    assert set(generated_ids) == {10, 12}
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert {record["id"] for record in records} == {10, 11, 12}


def test_exact_id_resume_rejects_a_malformed_nonfinal_line(tmp_path):
    output = tmp_path / "output.jsonl"
    first = {"id": 10, "status": "success"}
    last = {"id": 12, "status": "success"}
    output.write_bytes(
        (json.dumps(first) + "\n").encode()
        + b'{"id": 11\n'
        + (json.dumps(last) + "\n").encode()
    )

    with pytest.raises(json.JSONDecodeError):
        generate_train_data.load_success_ids(output)


def test_limited_exact_id_smoke_fails_when_any_selected_id_fails(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "smoke.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": source_id,
                    "conversations": [{"role": "user", "content": str(source_id)}],
                }
            )
            for source_id in (10, 11, 12)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generate_train_data, "validate_servers", lambda _args: ["worker"])

    def fake_call(_args, _server, sample, max_tokens=None):
        if sample["id"] == 11:
            sample["status"] = "error"
            sample["error"] = "injected smoke failure"
        else:
            sample["conversations"].append(
                {"role": "assistant", "content": "answer"}
            )
            sample["status"] = "success"
        return sample

    monkeypatch.setattr(generate_train_data, "call_sglang", fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_train_data.py",
            "--model",
            "target",
            "--server-address",
            "worker",
            "--input-file-path",
            str(source),
            "--output-file-path",
            str(output),
            "--num-samples",
            "2",
            "--resume",
            "--resume-by-id",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_train_data.main()

    assert exc_info.value.code == 1
    assert generate_train_data.load_success_ids(output) == {10}


def test_shard_then_validate_and_merge_official_conversations(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    shard_dir = tmp_path / "shards"
    generated_dir = tmp_path / "generated"
    merged = tmp_path / "merged.jsonl"
    source_records = [
        {
            "id": source_id,
            "conversations": [
                {"role": "user", "content": f"u{source_id}-1"},
                {"role": "assistant", "content": "old"},
                {"role": "user", "content": f"u{source_id}-2"},
                {"role": "assistant", "content": "old"},
            ],
        }
        for source_id in range(5)
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in source_records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_jsonl.py",
            "--input",
            str(source),
            "--output-dir",
            str(shard_dir),
            "--num-shards",
            "2",
            "--expected-total",
            "5",
        ],
    )
    shard_jsonl.main()
    assert [
        len(path.read_text().splitlines())
        for path in sorted(shard_dir.glob("shard_*.jsonl"))
    ] == [3, 2]
    manifest_path = shard_dir / "manifest.json"
    first_shard = sorted(shard_dir.glob("shard_*.jsonl"))[0]
    report = verify_shard(
        manifest_path=manifest_path,
        shard_path=first_shard,
        shard_index=0,
        num_shards=2,
        expected_total=5,
    )
    assert report["rows"] == 3

    original_bytes = first_shard.read_bytes()
    first_shard.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        verify_shard(
            manifest_path=manifest_path,
            shard_path=first_shard,
            shard_index=0,
            num_shards=2,
            expected_total=5,
        )
    first_shard.write_bytes(original_bytes)

    generated_dir.mkdir()
    for source_shard in sorted(shard_dir.glob("shard_*.jsonl")):
        regenerated = []
        for line in source_shard.read_text().splitlines():
            record = json.loads(line)
            users = [
                message
                for message in record["conversations"]
                if message["role"] == "user"
            ]
            conversations = []
            for user in users:
                conversations.extend(
                    [user, {"role": "assistant", "content": "new"}]
                )
            regenerated.append(
                {**record, "status": "success", "conversations": conversations}
            )
        (generated_dir / source_shard.name).write_text(
            "\n".join(json.dumps(record) for record in regenerated) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_regenerated_conversations.py",
            *[str(path) for path in sorted(generated_dir.glob("shard_*.jsonl"))],
            "--source-inputs",
            *[str(path) for path in sorted(shard_dir.glob("shard_*.jsonl"))],
            "--expected-total",
            "5",
            "--merge-out",
            str(merged),
        ],
    )
    validate_regenerated_conversations.main()
    assert len(merged.read_text().splitlines()) == 5


@pytest.mark.parametrize(
    ("model_key", "served_model", "sampling", "sampling_tag"),
    [
        ("qwen3_4b", "Qwen/Qwen3-4B", "0.7 0.8 20", "temp07"),
    ],
)
def test_launcher_preflight_uses_model_sampling_and_verified_shard(
    tmp_path, monkeypatch, model_key, served_model, sampling, sampling_tag
):
    source = tmp_path / "source.jsonl"
    shard_dir = tmp_path / "shards"
    model_dir = tmp_path / model_key
    model_dir.mkdir()
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": source_id,
                    "conversations": [
                        {"role": "user", "content": f"u{source_id}"}
                    ],
                }
            )
            for source_id in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_jsonl.py",
            "--input",
            str(source),
            "--output-dir",
            str(shard_dir),
            "--num-shards",
            "2",
            "--expected-total",
            "5",
        ],
    )
    shard_jsonl.main()

    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "REPO_ROOT": str(repo_root),
            "PYTHON_BIN": sys.executable,
            "SHARD_INPUT_DIR": str(shard_dir),
            "EXPECTED_TRAIN_ROWS": "5",
            "MODEL_PATH": str(model_dir),
            "SERVED_MODEL_NAME": served_model,
            "PREFLIGHT_ONLY": "1",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts/data/launch_rollout_shard.sh"),
            model_key,
            "0",
            "2",
            "0",
        ],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    temperature, top_p, top_k = sampling.split()
    assert f"PREFLIGHT OK model_key={model_key}" in result.stdout
    assert f"model_path={model_dir}" in result.stdout
    assert f"temperature={temperature}" in result.stdout
    assert f"top_p={top_p}" in result.stdout
    assert f"top_k={top_k}" in result.stdout
    assert f"multiturn_{sampling_tag}_max4096" in result.stdout


def test_online_token_conversion_filters_and_exactly_resumes(tmp_path, monkeypatch):
    source = tmp_path / "merged.jsonl"
    output = tmp_path / "online_train.jsonl"
    rejected = tmp_path / "online_rejected.jsonl"
    records = [
        {
            "id": 10,
            "conversations": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
        },
        {
            "id": 11,
            "conversations": [
                {"role": "user", "content": "short"},
                {"role": "assistant", "content": "x"},
            ],
        },
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tokenize_official_rollouts.AutoTokenizer,
        "from_pretrained",
        lambda _target: object(),
    )

    def fake_preprocess(*, record, tokenizer, chat_template, max_length):
        assert tokenizer is not None
        assert chat_template == "qwen"
        assert max_length == 4096
        loss_tokens = 14 if record["id"] == 10 else 1
        return {
            "input_ids": torch.arange(loss_tokens + 1),
            "loss_mask": torch.tensor([0] + [1] * loss_tokens),
        }

    monkeypatch.setattr(tokenize_official_rollouts, "preprocess_record", fake_preprocess)

    base_argv = [
        "tokenize_official_rollouts.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--rejected-output",
        str(rejected),
        "--target",
        "Qwen/Qwen3-4B",
        "--chat-template",
        "qwen",
        "--expected-total",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    tokenize_official_rollouts.main()

    accepted_rows = [json.loads(line) for line in output.read_text().splitlines()]
    rejected_rows = [json.loads(line) for line in rejected.read_text().splitlines()]
    assert [row["source_id"] for row in accepted_rows] == [10]
    assert accepted_rows[0]["source_user_turns"] == 2
    assert sum(accepted_rows[0]["loss_mask"]) == 14
    assert [row["source_id"] for row in rejected_rows] == [11]
    completion = json.loads(
        (tmp_path / "online_train.jsonl.complete.json").read_text()
    )
    assert completion["input_rows"] == 2
    assert completion["accepted_rows"] == 1
    assert completion["rejected_rows"] == 1

    with output.open("ab") as handle:
        handle.write(b'{"source_id": 999')
    with rejected.open("ab") as handle:
        handle.write(b'{"source_id": 998')
    monkeypatch.setattr(sys, "argv", [*base_argv, "--resume"])
    tokenize_official_rollouts.main()
    accepted_rows = [json.loads(line) for line in output.read_text().splitlines()]
    rejected_rows = [json.loads(line) for line in rejected.read_text().splitlines()]
    assert [row["source_id"] for row in accepted_rows] == [10]
    assert [row["source_id"] for row in rejected_rows] == [11]
