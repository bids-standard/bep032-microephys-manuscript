#!/usr/bin/env python
"""
Generate publication-ready probe geometry components for BEP032 Figure 3.

Produces individual SVG + PNG images designed for assembly in draw.io.
Three-colour scheme matches Figure 1 (ecephys overview):
    GREEN  (#388E3C) = probes.tsv   → probe outline / contour
    PINK   (#C2185B) = electrodes.tsv → contact positions & shapes
    ORANGE (#F57C00) = channels.tsv → device_channel_indices / wiring

Components:
    A. Annotated example probe (A4x8 NeuroNexus, 32 ch, 4 shanks)
       — custom 3-colour drawing (outline, contacts, IDs)
    B. Colour-coded JSON snippet matching the 3-colour scheme
    C. Diversity probe set (tetrode, linear, neuropixels, multi-shank)
    D. Reference 1×4 composite strip

Usage:
    conda run -n bep032-figures python generate_probe_plots.py

Output:
    figures/probes/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
import numpy as np
import probeinterface as pi
from probeinterface import library, plotting

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "probes"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Three-colour palette — matches Figure 1 (ecephys overview)
# ---------------------------------------------------------------------------
# GREEN family — probes.tsv: probe outline, contour, manufacturer, model
GREEN_FILL = "#E8F5E9"       # green 50 (light fill)
GREEN_EDGE = "#388E3C"       # green 700 (stroke)
GREEN_HEADER = "#C8E6C9"     # green 100 (header bg)

# PINK family — electrodes.tsv: contact positions, shapes, shank_ids
PINK_FILL = "#FCE4EC"        # pink 50 (light fill)
PINK_EDGE = "#C2185B"        # pink 700 (stroke)
PINK_HEADER = "#F8BBD9"      # pink 100 (header bg)

# ORANGE family — channels.tsv: contact IDs, device_channel_indices
ORANGE_FILL = "#FFF3E0"      # orange 50 (light fill)
ORANGE_EDGE = "#F57C00"      # orange 700 (stroke)
ORANGE_HEADER = "#FFE0B2"    # orange 100 (header bg)

# Neutral
LABEL_COLOR = "#212121"      # near-black
AXIS_COLOR = "#616161"       # gray 700
JSON_BG = "#F5F5F5"          # light gray

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.6,
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": AXIS_COLOR,
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# Shared ProbeInterface plotting kwargs (for diversity strip — default green)
PI_CONTACTS_KARGS = dict(alpha=1.0, edgecolor="#2E7D32", lw=0.4)
PI_PROBE_SHAPE_KWARGS = dict(facecolor="#C8E6C9", edgecolor="#66BB6A",
                             lw=0.8, alpha=0.6)


# ===================================================================
# Helpers
# ===================================================================

def style_axis(ax, xlabel="x (μm)", ylabel="y (μm)"):
    """Clean axes: only bottom/left spines, μm labels."""
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.tick_params(axis="both", labelsize=6, pad=2)


def save_fig(fig, filename):
    """Save figure as SVG + PNG to OUT_DIR."""
    for fmt in ("svg", "png"):
        out = OUT_DIR / f"{filename}.{fmt}"
        fig.savefig(out, format=fmt, dpi=300, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
        print(f"  {out}")
    plt.close(fig)


# ===================================================================
# A. Three-colour annotated probe — A4x8 NeuroNexus
# ===================================================================

def make_annotated_probe():
    """
    Custom 3-colour drawing of A4x8 multi-shank probe:
      GREEN  outline      → probe_planar_contour, annotations (probes.tsv)
      PINK   contacts+IDs → contact_positions, contact_shapes, contact_ids (electrodes.tsv)
      ORANGE channels     → device_channel_indices (channels.tsv)

    Uses raw matplotlib patches instead of plot_probe() for per-feature
    colour control.
    """
    print("\nA. Three-colour annotated probe — A4x8 (NeuroNexus, 32 ch)")
    probe = library.get_probe("neuronexus", "A4x8-5mm-100-200-177")
    probe.create_auto_shape(probe_type="tip")

    # Typical setup: Intan RHD2132 headstage via A32-OM32 Omnetics adapter.
    # Connector pin N → headstage input N-1 (standard 1:1 Omnetics mapping).
    # The interleaved contact_ids (1,8,2,7,...) then produce interleaved
    # device_channel_indices (0,7,1,6,...) reflecting the manufacturer wiring.
    dci = [int(cid) - 1 for cid in probe.contact_ids]
    probe.set_device_channel_indices(dci)

    fig, ax = plt.subplots(figsize=(5, 5))

    # --- 1. GREEN: Probe outline (probe_planar_contour → probes.tsv) ---
    contour = probe.probe_planar_contour
    outline = Polygon(
        contour, closed=True,
        facecolor=GREEN_FILL, edgecolor=GREEN_EDGE,
        lw=1.8, alpha=0.7, zorder=1,
    )
    ax.add_patch(outline)

    # --- 2. PINK: Contact positions, shapes & IDs (→ electrodes.tsv) ---
    #     contact_positions → x, y columns
    #     contact_shapes    → electrode_shape column
    #     contact_ids       → name column (electrode identifier)
    positions = probe.contact_positions
    radii = [p["radius"] for p in probe.contact_shape_params]

    for pos, r in zip(positions, radii):
        contact = Circle(
            pos, radius=r,
            facecolor=PINK_FILL, edgecolor=PINK_EDGE,
            lw=0.6, zorder=2,
        )
        ax.add_patch(contact)

    # contact_ids = electrode names (electrodes.tsv 'name' column)
    label_offset_x = 28  # μm to the right
    label_offset_y = 12  # μm above centre
    label_drop = -12      # μm shift both labels downward
    for pos, cid in zip(positions, probe.contact_ids):
        ax.text(
            pos[0] + label_offset_x, pos[1] + label_offset_y + label_drop,
            str(cid),
            color=PINK_EDGE, fontsize=8, fontweight="bold",
            ha="left", va="center", zorder=3,
        )

    # --- 3. ORANGE: Device channel indices (→ channels.tsv) ---
    #     device_channel_indices → maps electrodes to hardware channels
    #     Shown as "ch0", "ch1", ... below the electrode name
    for pos, ch_idx in zip(positions, probe.device_channel_indices):
        ax.text(
            pos[0] + label_offset_x, pos[1] - label_offset_y + label_drop,
            f"ch{ch_idx}",
            color=ORANGE_EDGE, fontsize=8, fontweight="bold",
            ha="left", va="center", zorder=3,
        )

    # --- Axis limits & styling ---
    x_pad, y_pad = 50, 60
    ax.set_xlim(contour[:, 0].min() - x_pad, contour[:, 0].max() + x_pad)
    ax.set_ylim(contour[:, 1].min() - y_pad, contour[:, 1].max() + y_pad)
    style_axis(ax)

    ax.set_title(
        "A4x8-5mm-100-200-177 (NeuroNexus)\n32 channels · 4 shanks",
        fontsize=9, fontweight="bold", color=LABEL_COLOR, pad=8,
    )

    # --- Colour legend ---
    # legend_handles = [
    #     mpatches.Patch(facecolor=GREEN_FILL, edgecolor=GREEN_EDGE, lw=1.2,
    #                    label="probe outline (probes.tsv)"),
    #     mpatches.Patch(facecolor=PINK_FILL, edgecolor=PINK_EDGE, lw=1.2,
    #                    label="contacts + IDs (electrodes.tsv)"),
    #     mpatches.Patch(facecolor=ORANGE_FILL, edgecolor=ORANGE_EDGE, lw=1.2,
    #                    label="channel indices (channels.tsv)"),
    # ]
    # ax.legend(
    #     handles=legend_handles, loc="upper right", fontsize=6,
    #     framealpha=0.9, edgecolor="#BDBDBD",
    # )

    save_fig(fig, "annotated_A4x8_coloured")
    return probe


# ===================================================================
# B. Colour-coded JSON snippet (3-colour scheme)
# ===================================================================

def make_json_snippet():
    """
    ProbeInterface JSON with fields coloured by BIDS level:
      GREEN  → probe-level: probe_planar_contour, annotations, si_units
      PINK   → electrode-level: contact_positions, contact_shapes, contact_ids, shank_ids
      ORANGE → channel-level: device_channel_indices
    """
    print("\nB. JSON snippet (3-colour coded)")

    json_lines = [
        ('{',                                    LABEL_COLOR),
        ('  "specification": "probeinterface",',  LABEL_COLOR),
        ('  "probes": [{',                       LABEL_COLOR),
        ('    "si_units": "um",',                GREEN_EDGE),
        ('    "annotations": {',                 GREEN_EDGE),
        ('      "manufacturer": "neuronexus",',  GREEN_EDGE),
        ('      "model_name": "A4x8-5mm-..."',  GREEN_EDGE),
        ('    },',                               GREEN_EDGE),
        ('    "contact_positions": [',           PINK_EDGE),
        ('      [0, 0], [0, 100],',             PINK_EDGE),
        ('      [200, 0], [200, 100], ...',      PINK_EDGE),
        ('    ],',                               PINK_EDGE),
        ('    "contact_shapes": ["circle"],',    PINK_EDGE),
        ('    "contact_shape_params": [',        PINK_EDGE),
        ('      {"radius": 7.5}',               PINK_EDGE),
        ('    ],',                               PINK_EDGE),
        ('    "contact_ids": ["1","8",...],',     PINK_EDGE),
        ('    "shank_ids": ["0","0",...],',       PINK_EDGE),
        ('    "probe_planar_contour": [',        GREEN_EDGE),
        ('      [-20, 720], [-20, -20],',         GREEN_EDGE),
        ('      [620, 720], ...',                GREEN_EDGE),
        ('    ],',                               GREEN_EDGE),
        ('    "device_channel_indices": [',      ORANGE_EDGE),
        ('      0, 7, 1, 6, 2, 5, ...',         ORANGE_EDGE),
        ('    ]',                                ORANGE_EDGE),
        ('  }]',                                 LABEL_COLOR),
        ('}',                                    LABEL_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(3.8, 5.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Background rectangle
    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor=JSON_BG, edgecolor="#BDBDBD", lw=0.8,
    )
    ax.add_patch(bg)

    # Render each line
    n = len(json_lines)
    line_height = 0.86 / n
    y_top = 0.95

    for i, (text, color) in enumerate(json_lines):
        y = y_top - i * line_height
        ax.text(0.06, y, text, fontfamily="monospace", fontsize=6,
                color=color, va="top", ha="left",
                transform=ax.transAxes)

    # Colour legend at bottom
    legend_y = 0.015
    legend_items = [
        (GREEN_EDGE, "probes.tsv"),
        (PINK_EDGE, "electrodes.tsv"),
        (ORANGE_EDGE, "channels.tsv"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 0.06 + i * 0.32
        ax.plot(x, legend_y, "s", color=color, markersize=5,
                transform=ax.transAxes)
        ax.text(x + 0.025, legend_y, label, fontsize=5.5, color=color,
                va="center", ha="left", transform=ax.transAxes,
                fontweight="bold")

    save_fig(fig, "json_snippet_coloured")


# ===================================================================
# C. Diversity probe set — individual plots (GREEN outline + PINK contacts)
# ===================================================================

def plot_probe_styled(probe, ax, with_contact_id=False, id_fontsize=5):
    """Plot a probe with GREEN outline (probes.tsv) + PINK contacts (electrodes.tsv)."""
    # GREEN: probe outline
    contour = probe.probe_planar_contour
    outline = Polygon(
        contour, closed=True,
        facecolor=GREEN_FILL, edgecolor=GREEN_EDGE,
        lw=1.2, alpha=0.7, zorder=1,
    )
    ax.add_patch(outline)

    # PINK: contact positions (handles circle, square, rect shapes)
    positions = probe.contact_positions
    for pos, shape, params in zip(positions, probe.contact_shapes,
                                   probe.contact_shape_params):
        if shape == "circle":
            patch = Circle(pos, radius=params["radius"],
                           facecolor=PINK_FILL, edgecolor=PINK_EDGE,
                           lw=0.4, zorder=2)
        else:  # square or rect
            w = params["width"]
            h = params.get("height", w)
            patch = plt.Rectangle((pos[0] - w / 2, pos[1] - h / 2), w, h,
                                  facecolor=PINK_FILL, edgecolor=PINK_EDGE,
                                  lw=0.4, zorder=2)
        ax.add_patch(patch)

    # Optional contact IDs (PINK) — only if the probe has them
    if with_contact_id and probe.contact_ids is not None:
        for pos, cid in zip(positions, probe.contact_ids):
            ax.text(
                pos[0], pos[1], str(cid),
                color=PINK_EDGE, fontsize=id_fontsize, fontweight="bold",
                ha="center", va="center", zorder=3,
            )

    # Set axis limits from contour
    x_pad, y_pad = 30, 30
    ax.set_xlim(contour[:, 0].min() - x_pad, contour[:, 0].max() + x_pad)
    ax.set_ylim(contour[:, 1].min() - y_pad, contour[:, 1].max() + y_pad)


def make_strip_probe(probe, title, filename, figsize=(2.0, 3.0),
                     zoom=None, with_contact_id=False, id_fontsize=5):
    """Single diversity-strip cell with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_probe_styled(probe, ax, with_contact_id=with_contact_id,
                      id_fontsize=id_fontsize)

    if zoom is not None:
        contour = probe.probe_planar_contour
        x_pad = zoom.get("x_pad", 40)
        ax.set_xlim(contour[:, 0].min() - x_pad,
                    contour[:, 0].max() + x_pad)
        if "ymin" in zoom:
            ax.set_ylim(zoom["ymin"], zoom["ymax"])

    style_axis(ax)
    ax.set_title(title, fontsize=8, fontweight="bold",
                 color=LABEL_COLOR, pad=6)

    save_fig(fig, filename)


def make_diversity_set():
    """Generate all 4 diversity probes as individual images."""

    # 1. Tetrode
    print("\nC1. Tetrode (4 ch)")
    p_tet = pi.generate_tetrode()
    p_tet.create_auto_shape(probe_type="tip")
    make_strip_probe(p_tet, "Tetrode\n4 channels",
                     "strip_tetrode", figsize=(1.8, 2.5),
                     with_contact_id=True, id_fontsize=6)

    # 2. Linear — Cambridge NeuroTech ASSY-77-H5
    print("\nC2. Linear — ASSY-77-H5 (64 ch)")
    p_lin = library.get_probe("cambridgeneurotech", "ASSY-77-H5")
    p_lin.create_auto_shape(probe_type="tip")
    pos = p_lin.contact_positions
    make_strip_probe(p_lin, "ASSY-77-H5\n64 ch · linear",
                     "strip_linear", figsize=(1.8, 3.5),
                     zoom={"x_pad": 30,
                           "ymin": pos[:, 1].min() - 30,
                           "ymax": pos[:, 1].min() + 300})

    # 3. Neuropixels — IMEC NP1000
    print("\nC3. Neuropixels — NP1000 (960 ch)")
    p_np = library.get_probe("imec", "NP1000")
    p_np.create_auto_shape(probe_type="tip")
    np_pos = p_np.contact_positions
    make_strip_probe(p_np, "Neuropixels 1.0\n960 ch · high-density",
                     "strip_neuropixels", figsize=(1.8, 3.5),
                     zoom={"x_pad": 30,
                           "ymin": np_pos[:, 1].min() - 30,
                           "ymax": np_pos[:, 1].min() + 400})

    # 4. Multi-shank — NeuroNexus A4x8
    print("\nC4. Multi-shank — A4x8 (32 ch)")
    p_ms = library.get_probe("neuronexus", "A4x8-5mm-100-200-177")
    p_ms.create_auto_shape(probe_type="tip")
    ms_pos = p_ms.contact_positions
    make_strip_probe(p_ms, "A4x8 (NeuroNexus)\n32 ch · 4 shanks",
                     "strip_multishank", figsize=(2.5, 3.5),
                     zoom={"x_pad": 50,
                           "ymin": ms_pos[:, 1].min() - 40,
                           "ymax": ms_pos[:, 1].max() + 60},
                     with_contact_id=True, id_fontsize=5)

    return p_tet, p_lin, p_np, p_ms


# ===================================================================
# D. Reference composite strip (1×4)
# ===================================================================

def make_composite_strip(probes_info):
    """
    1×4 row of all diversity probes for reference.

    Parameters
    ----------
    probes_info : list of (probe, title, zoom_dict_or_None)
    """
    print("\nD. Composite diversity strip (1×4)")
    n = len(probes_info)
    fig, axes = plt.subplots(1, n, figsize=(8, 3.5))

    for ax, (probe, title, zoom) in zip(axes, probes_info):
        plot_probe_styled(probe, ax, with_contact_id=False)
        style_axis(ax)
        ax.set_title(title, fontsize=7, fontweight="bold",
                     color=LABEL_COLOR, pad=4)

        if zoom is not None:
            contour = probe.probe_planar_contour
            x_pad = zoom.get("x_pad", 40)
            ax.set_xlim(contour[:, 0].min() - x_pad,
                        contour[:, 0].max() + x_pad)
            if "ymin" in zoom:
                ax.set_ylim(zoom["ymin"], zoom["ymax"])

    fig.tight_layout(pad=0.8)
    save_fig(fig, "diversity_strip")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print(f"ProbeInterface version: {pi.__version__}")
    print(f"Output directory: {OUT_DIR}\n")

    # A. Three-colour annotated centrepiece
    p_annotated = make_annotated_probe()

    # B. Colour-coded JSON snippet
    make_json_snippet()

    # C. Individual diversity probes
    p_tet, p_lin, p_np, p_ms = make_diversity_set()

    # D. Composite reference
    np_pos = p_np.contact_positions
    ms_pos = p_ms.contact_positions
    lin_pos = p_lin.contact_positions

    make_composite_strip([
        (p_tet, "Tetrode\n(4 ch)", None),
        (p_lin, "ASSY-77-H5\n(64 ch)",
         {"x_pad": 30,
          "ymin": lin_pos[:, 1].min() - 30,
          "ymax": lin_pos[:, 1].min() + 300}),
        (p_np, "Neuropixels 1.0\n(960 ch)",
         {"x_pad": 30,
          "ymin": np_pos[:, 1].min() - 30,
          "ymax": np_pos[:, 1].min() + 400}),
        (p_ms, "A4x8 NeuroNexus\n(32 ch, 4 shanks)",
         {"x_pad": 50,
          "ymin": ms_pos[:, 1].min() - 40,
          "ymax": ms_pos[:, 1].max() + 60}),
    ])

    print(f"\nDone! {len(list(OUT_DIR.glob('*')))} files in {OUT_DIR}")
