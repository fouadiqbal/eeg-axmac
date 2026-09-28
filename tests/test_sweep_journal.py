import ast
import json
import os
from pathlib import Path


def _journal_functions():
    runner = Path(__file__).parents[1] / "notebooks" / "eegmmidb_fullfold_sensitivity_kaggle.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    names = {"read_case_journal", "append_case_journal"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"json": json, "os": os}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(runner), "exec"), namespace)
    return namespace["read_case_journal"], namespace["append_case_journal"]


def _record(multiplier_id: str) -> dict:
    return {
        "layer": "temporal",
        "multiplier_id": multiplier_id,
        "result": {"delta_acc": -0.01},
        "subject_rows": [{"subject": 1, "delta_acc": -0.02}],
    }


def test_case_journal_recovers_after_interrupted_append(tmp_path):
    read_case_journal, append_case_journal = _journal_functions()
    path = tmp_path / "fold0.jsonl"
    append_case_journal(path, _record("mul_a"))
    # Simulate process termination halfway through writing the next JSON record.
    with path.open("ab") as stream:
        stream.write(b'{"layer":"spatial","multiplier_id":"mul_b"')

    recovered = read_case_journal(path)
    assert [(r["layer"], r["multiplier_id"]) for r in recovered] == [("temporal", "mul_a")]
    assert path.read_bytes().endswith(b"\n")

    append_case_journal(path, _record("mul_b"))
    completed = read_case_journal(path)
    assert [r["multiplier_id"] for r in completed] == ["mul_a", "mul_b"]
    assert json.loads(path.read_text().splitlines()[1])["result"]["delta_acc"] == -0.01


def test_case_journal_rejects_duplicate_completed_cases(tmp_path):
    read_case_journal, append_case_journal = _journal_functions()
    path = tmp_path / "fold0.jsonl"
    append_case_journal(path, _record("mul_a"))
    append_case_journal(path, _record("mul_a"))
    try:
        read_case_journal(path)
    except ValueError as exc:
        assert "Duplicate case keys" in str(exc)
    else:
        raise AssertionError("duplicate journal case was accepted")
