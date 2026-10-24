"""reports/tables/fold_scores.csv (the source of results/, notebook 02 and the paper's tables) matches the committed raw
result JSONs, row by row."""
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))


def test_every_fold_score_row_matches_its_raw_result():
    import check_fold_scores
    assert check_fold_scores.main() == []
