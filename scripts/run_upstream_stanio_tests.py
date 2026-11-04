"""Run the *unchanged* official stan-dev/stanio test suite on staniorust.

This intentionally does not vendor BSD-3-Clause tests or fixtures. Clone the
upstream repository before running, then alias its import paths in memory:

    git clone --depth 1 https://github.com/stan-dev/stanio.git upstream-stanio
    python scripts/run_upstream_stanio_tests.py upstream-stanio

Any upstream failures are real and cause a non-zero exit code. The Rust
extension must already be installed (e.g. with ``maturin develop --release``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Cloned stan-dev/stanio repository")
    arguments, extra_args = parser.parse_known_args()
    test_dir = arguments.upstream.resolve() / "test"
    required = ["test_csv.py", "test_json.py", "test_reshape.py"]
    missing = [name for name in required if not (test_dir / name).is_file()]
    if missing:
        parser.error(f"upstream StanIO tests not found: {', '.join(missing)}")
    import staniorust
    import staniorust.csv
    import staniorust.json
    import staniorust.reshape

    # Make direct imports from StanIO's unchanged tests resolve to our API.
    sys.modules["stanio"] = staniorust
    sys.modules["stanio.csv"] = staniorust.csv
    sys.modules["stanio.json"] = staniorust.json
    sys.modules["stanio.reshape"] = staniorust.reshape
    return int(pytest.main(["-q", str(test_dir), *extra_args]))


if __name__ == "__main__":
    sys.exit(main())
