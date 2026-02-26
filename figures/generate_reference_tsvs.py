#!/usr/bin/env python
"""
Generate reference BEP032 TSVs and ProbeInterface JSON from the A4x8 probe.

This creates ground-truth files that inform the drawio figure tables.
Run with:  conda run -n bep032-figures python generate_reference_tsvs.py

Output:
    figures/probes/reference/
        probes.tsv
        electrodes.tsv
        channels.tsv
        A4x8_probeinterface.json
"""

from pathlib import Path
import json
import csv

import numpy as np
from probeinterface import library

OUT_DIR = Path(__file__).parent / "probes" / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load and configure probe ---
probe = library.get_probe("neuronexus", "A4x8-5mm-100-200-177")
probe.create_auto_shape(probe_type="tip")

# Typical setup: Intan RHD2132 headstage via A32-OM32 Omnetics adapter.
# Connector pin N → headstage input N-1 (standard 1:1 Omnetics mapping).
# The interleaved contact_ids (1,8,2,7,...) then produce interleaved
# device_channel_indices (0,7,1,6,...) reflecting the manufacturer wiring.
dci = [int(cid) - 1 for cid in probe.contact_ids]
probe.set_device_channel_indices(dci)

n = probe.get_contact_count()

# --- probes.tsv ---
with open(OUT_DIR / "probes.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["probe_name", "type", "manufacturer", "model"])
    w.writerow(["A4x8", "silicon", "neuronexus", "A4x8-5mm-100-200-177"])

# --- electrodes.tsv ---
with open(OUT_DIR / "electrodes.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["name", "probe_name", "x", "y", "z", "shank_id",
                "electrode_shape"])
    for i in range(n):
        cid = probe.contact_ids[i]
        x, y = probe.contact_positions[i]
        sid = probe.shank_ids[i]
        shape = probe.contact_shapes[i]
        w.writerow([cid, "A4x8", f"{x:.1f}", f"{y:.1f}", "0.0", sid, shape])

# --- channels.tsv ---
with open(OUT_DIR / "channels.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["name", "electrode_name", "type", "units", "sampling_frequency"])
    for i in range(n):
        ch_idx = probe.device_channel_indices[i]
        cid = probe.contact_ids[i]
        w.writerow([f"ch{ch_idx}", cid, "BB", "uV", "30000"])

# --- ProbeInterface JSON ---
d = probe.to_dict()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.str_):
            return str(obj)
        return super().default(obj)


with open(OUT_DIR / "A4x8_probeinterface.json", "w") as f:
    json.dump({"specification": "probeinterface", "probes": [d]},
              f, indent=2, cls=NumpyEncoder)

# --- Summary ---
print(f"Output: {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")

print("\n--- electrodes.tsv (first 3 per shank) ---")
for sid in sorted(set(probe.shank_ids)):
    mask = np.array(probe.shank_ids) == sid
    idxs = np.where(mask)[0][:3]
    for i in idxs:
        cid = probe.contact_ids[i]
        x, y = probe.contact_positions[i]
        dci = probe.device_channel_indices[i]
        print(f"  name={cid:>2s}  probe=A4x8  x={x:5.0f}  y={y:5.0f}"
              f"  shank={sid}  → ch{dci}")
