"""Static contracts for the independent five-sleeper realization campaign."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "mesonet" / "track_ensemble_seeds_v1.json"
GEN = ROOT / "mesonet" / "generate_track_ensemble_hedy.slurm"
SEG = ROOT / "mesonet" / "segment_track_ensemble_hedy.slurm"
FINAL = ROOT / "mesonet" / "finalize_track_ensemble_handoff_hedy.slurm"
VERIFY = ROOT / "mesonet" / "verify_track_ensemble_hedy.slurm"


def test_campaign_uses_distinct_seeds_and_historical_reference():
    manifest = json.loads(SEEDS.read_text())
    assert manifest["seeds"] == [25101, 25102, 25103]
    assert len(set(manifest["seeds"])) == 3
    assert manifest["historical_reference"]["seed"] == 25100
    assert len(manifest["historical_reference"]["profile_sha256"]) == 64
    assert len(manifest["historical_reference"]["polyhedra_sha256"]) == 64


def test_campaign_stages_are_gpu_only_and_fail_closed():
    for path in (GEN, SEG, FINAL, VERIFY):
        text = path.read_text()
        assert "#SBATCH --partition=gpu" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "set -euo pipefail" in text
        assert "CODE_ARCHIVE_SHA256" in text
        assert "SEED_MANIFEST_SHA256" in text
    for path in (FINAL, VERIFY):
        text = path.read_text()
        assert "SOURCE_CODE_ARCHIVE_SHA256" in text
        assert "FINALIZER_CODE_ARCHIVE_SHA256" in text


def test_finalizer_applies_centroid_scale_and_verifier_uses_handoff_only():
    final = FINAL.read_text()
    verify = VERIFY.read_text()
    assert "corrected=center+scale*(vertices-center)" in final
    assert "float(grain['volume'])*scale**3" in final
    assert "0.999<=ratio<=1.001" in final
    assert "mechanical_preflight_input_ready_not_mechanically_qualified" in final
    assert "handoff/physical_corrected.json" in verify
    assert "corrected_polyhedra_sha256" in verify
    assert "source_code_archive_sha256" in verify
    assert "finalizer_code_archive_sha256" in verify
    assert "reference[field] in values" in verify
    assert "mechanically_qualified':False" in verify
