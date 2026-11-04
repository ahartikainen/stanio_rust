"""Small, deterministic CmdStan-like fixtures with adaptation comments."""

from pathlib import Path

import pytest


@pytest.fixture
def csv_files(tmp_path: Path) -> list[Path]:
    """Return two two-draw chains containing one saved warmup draw each."""
    paths = []
    for chain in (1, 2):
        path = tmp_path / f"chain-{chain}.csv"
        path.write_text(
            "# stan_version_major = 2\n"
            "# method = sample (Default)\n"
            "# num_warmup = 1\n"
            "# num_samples = 2\n"
            "# thin = 1\n"
            "# save_warmup = 1\n"
            "# id = 1\n"
            "lp__,alpha,beta.1,beta.2,mat.1.1,mat.1.2,mat.2.1,mat.2.2\n"
            f"-5,0,10,20,11,12,21,22\n"
            "# Adaptation terminated\n"
            "# Step size = 0.8\n"
            "# Diagonal elements of inverse mass matrix:\n"
            "# 1.0, 2.0\n"
            f"-1,{chain},1,2,1,2,3,4\n"
            f"-2,{chain + 1},3,4,5,6,7,8\n"
            "# Elapsed Time: 0.4 seconds (Sampling)\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    path = tmp_path / "inputs.json"
    path.write_text(
        '{"N":3,"is_ready":true,"x":[[1,2],[3,4]],'
        '"y":[0.1,0.2],"name":"example",'
        '"other":{"a":[3.0,4.0]}}',
        encoding="utf-8",
    )
    return path
