#!/usr/bin/env bash
# E52 on a GPU machine: end-to-end tests, then the dev screen of cte_depth_top3_attn on the P12 protocol.
# The baseline (cte_depth_top3) rows of the same splits are E51's (~/e51/dev_fold*.csv); caches are shared (~/e51/cache).
set -u
cd ~/text-tabular-nlp
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p ~/e52
.venv/bin/python -m pytest -q tests/test_attn.py tests/test_depth.py -k "not packaged_featurizer" > ~/e52/tests.txt 2>&1
tail -3 ~/e52/tests.txt
if ! tail -1 ~/e52/tests.txt | grep -q "passed" || tail -1 ~/e52/tests.txt | grep -q "failed"; then
  echo "TESTS FAILED - not running the screen" > ~/e52/DONE; exit 1
fi
DS="BIN_TEXT_JIGSAW_TOXICITY BIN_TEXT_KICKSTARTER_FUNDING MUL_TEXT_PRODUCT_SENTIMENT REG_TEXT_BOOK_PRICE REG_TEXT_BABIES_PRICES REG_TEXT_VIDEO_GAMES_SALES MUL_TEXT_WOMEN_CLOTHING_REVIEW REG_TEXT_VANCOUVER_SALARIES REG_TEXT_MONTGOMERY_SALARIES"
for f in 0 1 2; do
  .venv/bin/python scripts/dev_eval.py --datasets $DS --methods cte_depth_top3_attn \
    --seeds 0 1 --fold $f --out ~/e52/dev_fold$f.csv --cache_dir ~/e51/cache > ~/e52/log_fold$f.txt 2>&1 &
done
wait
echo "E52 DONE" > ~/e52/DONE
