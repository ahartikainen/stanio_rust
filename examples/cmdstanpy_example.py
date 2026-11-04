"""Reading the output of a previously completed CmdStanPy sampling run.

Start from a fitted object named ``fit`` in a Jupyter notebook.
"""

import staniorust as sio

# fit = CmdStanModel(...).sample(...)
paths = fit.runset.csv_files  # noqa: F821

header, values = sio.read_csv(paths, columns=["lp__", "theta"], threads=8)
print(header, values.shape)

# Reconstruct parameter arrays, using Stan 1-based indices.
arrays = sio.read_csv(paths, columns=["theta"], reshape=True)
print({key: array.shape for key, array in arrays.items()})

# A random row range never decodes the other draws.
small = sio.read_csv(paths, rows=slice(100, 200), columns=["theta"])

# Compute statistics with no all-draw array allocation.
summary = sio.summarize_csv(paths, columns=["theta"], quantiles=[0.05, 0.5, 0.95])
print(summary)

# Return one chain/phase per chunk.
for block in sio.iter_csv(paths, chunksize=10_000, columns=["theta"]):
    print(block.chain, block.draw_start, block.data.shape)
