import pathlib
import sys
import types

sys.modules.setdefault("numba", types.SimpleNamespace(njit=lambda f: f))

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "TinyRecursiveModels"))

from TinyRecursiveModels.evaluators.arc_utils import score_candidates


def test_score_candidates_orders_by_average_then_count():
    candidates = {
        "hash_low": [4, 1.2],   # avg = 0.3
        "hash_mid": [3, 1.5],   # avg = 0.5
        "hash_high": [1, 0.7],  # avg = 0.7
    }

    ranked = score_candidates(candidates)
    assert [h for h, _, _ in ranked] == ["hash_high", "hash_mid", "hash_low"]


def test_score_candidates_breaks_ties_with_count():
    candidates = {
        "hash_many": [4, 2.0],   # avg = 0.5 count 4
        "hash_few": [1, 0.5],    # avg = 0.5 count 1
        "hash_lower": [2, 0.6],  # avg = 0.3
    }

    ranked = score_candidates(candidates)
    assert [h for h, _, _ in ranked] == ["hash_many", "hash_few", "hash_lower"]
