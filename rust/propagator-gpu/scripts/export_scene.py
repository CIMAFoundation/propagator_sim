"""Export the docs demo landscape and a fuel YAML into the plain files
`gpu_bench` reads with `SCENE=<dir>` (header.txt, veg.i32, dem.f32,
fuels.txt), so the GPU and CPU cores can be compared natively on real data.

    uv run python rust/propagator-gpu/scripts/export_scene.py /tmp/scene
"""

from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / "docs/demo/alexandroupolis.bin.gz"
FUELS = ROOT / "example/pedrogao/fuels_eu12.yaml"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    buf = gzip.open(ASSET).read()
    rows, cols = struct.unpack_from("<II", buf, 0)
    cell = struct.unpack_from("<f", buf, 8)[0]
    ign_row, ign_col = struct.unpack_from("<ii", buf, 12)
    n = rows * cols
    pad = n % 2
    veg = np.frombuffer(buf, dtype=np.int8, count=n, offset=20).astype(
        np.int32
    )
    dem = np.frombuffer(buf, dtype="<i2", count=n, offset=20 + n + pad).astype(
        np.float32
    )
    veg.tofile(out / "veg.i32")
    dem.tofile(out / "dem.f32")
    (out / "header.txt").write_text(
        f"{rows} {cols} {cell} {ign_row} {ign_col}\n"
    )

    fuels = yaml.safe_load(FUELS.read_text())["fuels"]
    lines = []
    for fid, f in fuels.items():
        spread = f.get("spread_probability", {})
        humidity = f.get("humidity")
        fields = [
            fid,
            f["v0"],
            f.get("d0", 0),
            f.get("d1", 0),
            f["hhv"],
            "nan" if humidity is None else humidity,
            int(bool(f.get("spotting", False))),
            f.get("prob_ign_by_embers", 0),
            int(bool(f.get("burn", True))),
            len(spread),
        ]
        fields += [f"{k}:{v}" for k, v in spread.items()]
        lines.append(" ".join(str(v) for v in fields))
    (out / "fuels.txt").write_text("\n".join(lines) + "\n")
    print(
        f"{rows}x{cols} cells at {cell} m, ignition ({ign_row}, {ign_col}), {len(fuels)} fuels -> {out}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "scene"))
