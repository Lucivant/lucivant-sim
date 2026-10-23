"""The notebook's in-browser copy of Orrery (apps/studio/public/pyodide/lucivant_sim.zip) matches
this source: rebuild it with `python apps/studio/scripts/build_sim_zip.py` after a change."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "apps" / "studio" / "public" / "pyodide" / "lucivant_sim.zip"


def test_the_browser_bundle_matches_the_source(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_sim_zip", ROOT / "apps" / "studio" / "scripts" / "build_sim_zip.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fresh = module.build(tmp_path / "fresh.zip")
    with zipfile.ZipFile(BUNDLE) as committed, zipfile.ZipFile(fresh) as rebuilt:
        assert committed.namelist() == rebuilt.namelist()
        for name in rebuilt.namelist():
            assert committed.read(name) == rebuilt.read(name), f"{name} changed: rebuild the browser bundle"
