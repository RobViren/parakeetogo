"""parity.py REF_DIR CPP_DIR: one line per stage: name, shape match, relative RMS error, max abs error."""
import sys
from pathlib import Path

import numpy as np

ref_dir, cpp_dir = Path(sys.argv[1]), Path(sys.argv[2])
for path in sorted(ref_dir.glob("*.npy")):
    other = cpp_dir / path.name
    if not other.exists():
        print(f"{path.stem}\tmissing")
        continue
    a, b = np.load(path).astype(np.float64), np.load(other).astype(np.float64)
    if a.shape != b.shape:
        print(f"{path.stem}\tshape {a.shape} vs {b.shape}")
        continue
    rms = np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a**2)), 1e-30) if a.size else 0.0
    print(f"{path.stem}\tok\t{rms:.2e}\t{np.abs(a - b).max() if a.size else 0.0:.2e}")
