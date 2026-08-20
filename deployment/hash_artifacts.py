#!/usr/bin/env python3
"""Pin artifact digests into deployment/model_manifest.json (spec section 3).

Run once after placing the three model files under models/. Every artifact
must exist; the reported sha256 is the value the warm-up fingerprint and the
startup verification compare against. Fails with a non-zero exit if any
artifact is missing or unreadable.
"""

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "deployment", "model_manifest.json")
CHUNK = 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    artifacts = manifest["artifacts"]
    failures = 0
    for name, info in sorted(artifacts.items()):
        path = os.path.join(REPO_ROOT, info["path"])
        if not os.path.isfile(path):
            print("MISSING  {} -> {}".format(name, info["path"]))
            failures += 1
            continue
        digest = sha256(path)
        info["sha256"] = digest
        print("PINNED   {} {} {}".format(name, digest[:16], info.get("dtype", "")))
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    if failures:
        print("{} artifact(s) missing; digests not pinned for them.".format(failures))
        return 1
    print("All artifact digests pinned in deployment/model_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())