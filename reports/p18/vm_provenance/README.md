# Provenance of the Tesla T4 results of P18

Made on the T4 machine on 2026-10-24 in its copy of the repository (HEAD `8b3bb76`, never pushed; `c63c26e` fetched from
GitHub), with:

```bash
for f in $(git ls-tree -r --name-only c63c26e -- multabench scripts/run_experiment.py jobs/p18.txt pyproject.toml uv.lock); do
  echo "$f $(git rev-parse c63c26e:$f) $(git hash-object "$f") $(tr -d '\r' < "$f" | git hash-object --stdin)"
done > vm_hash_manifest.txt
```

- `vm_hash_manifest.txt`: per file, the blob in `c63c26e`, the hash of the machine's file, and its hash with CR removed.
  All 485 files: 481 byte-identical, 4 identical after removing CR (`jobs/p18.txt`, `depth.py`, `variants.py`,
  `run_experiment.py`, copied from a Windows checkout). Rule: column 2 = column 3 or column 2 = column 4 (38 upstream
  benchmark files contain CR bytes in `c63c26e` itself, 36 CSVs with CRLF line endings and 2 PNG images, so for them
  only column 3 matches; all 38 are byte-identical).
- `vm_file_times.txt`: each file's last modification time on the machine. The newest is `multabench/ours/depth.py`,
  2026-10-23 20:12:54 +0300 (the `f145a76` → `c63c26e` change: token states kept in memory, not written to disk).
- `../provenance_t4.csv`: each result's start and end time: the 35 LightGBM runs that started before that change (last
  start 19:59:57) ran `f145a76`, the other 265 (first start 20:13:24) the `c63c26e` content.
