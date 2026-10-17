from __future__ import annotations

import importlib.util
import json
import tarfile

from forensics.config import PROJECT_ROOT


def load_seed(monkeypatch, snapshot):
    spec = importlib.util.spec_from_file_location("seed_demo", PROJECT_ROOT / "scripts" / "seed_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SNAPSHOT", snapshot)
    return module


def test_export_scrubs_local_paths_and_unpack_restores(tmp_path, monkeypatch):
    seed = load_seed(monkeypatch, tmp_path / "demo.tar.gz")
    src = tmp_path / "src"
    (src / "diagnoses").mkdir(parents=True)
    trace = {"trace_id": "t1", "source_path": f"{PROJECT_ROOT}/data/documents/a.md"}
    (src / "t1.json").write_text(json.dumps(trace))
    (src / "diagnoses" / "t1.json").write_text("{}")

    assert seed.export(src) == 0
    with tarfile.open(tmp_path / "demo.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["diagnoses/t1.json", "t1.json"]

    dst = tmp_path / "dst"
    assert seed.unpack(dst, force=False) == 0
    assert json.loads((dst / "t1.json").read_text())["source_path"] == "data/documents/a.md"
    (dst / "t1.json").write_text("{}")
    assert seed.unpack(dst, force=False) == 0  # existing traces are left alone
    assert (dst / "t1.json").read_text() == "{}"


def test_committed_snapshot_has_no_local_paths():
    snapshot = PROJECT_ROOT / "demo" / "demo_traces.tar.gz"
    with tarfile.open(snapshot) as tar:
        for member in tar.getmembers():
            if member.name.endswith(".json"):
                assert b"/Users/" not in tar.extractfile(member).read(), member.name
