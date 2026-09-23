# splits/

Generated once by `pipeline.data.build_manifest` (see `../instructions.md`,
step 6). Fixes the leakage bugs in the legacy notebook by defining the
dev_train / dev_val / final_test partition exactly once and reusing it
everywhere.

| File | Contents |
|---|---|
| `manifest.csv` | filepath, label, class_idx, split, file_hash for every image |
| `classes.json` | sorted list of class names, index = position (matches `class_idx`) |
| `duplicate_report.txt` | byte-identical duplicate images found and reassigned to prevent leakage |
