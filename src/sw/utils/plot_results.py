import csv, os, warnings
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Font / style ──────────────────────────────────────────────────────────────
FONT_BASE   = 8
FONT_AXES   = 8
FONT_TITLE  = 8
FONT_LEGEND = 7
FONT_TICKS  = 7

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif"],
    "text.usetex":     False,
    "svg.fonttype":    "none",
    "font.size":       FONT_BASE,
    "axes.labelsize":  FONT_AXES,
    "axes.titlesize":  FONT_TITLE,
    "legend.fontsize": FONT_LEGEND,
    "xtick.labelsize": FONT_TICKS,
    "ytick.labelsize": FONT_TICKS,
    "lines.linewidth": 1,
    "axes.grid":       True,
    "grid.linewidth":  0.3,
    "figure.dpi":      300,
    "savefig.dpi":     300,
})

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_COLS = ["FalinCNN", "Falin_XGB", "MobileNetV2", "MobileNet_XGB"]

PALETTE = {
    "FalinCNN":      "#4C72B0",
    "Falin_XGB":     "#55A868",
    "MobileNetV2":   "#DD8452",
    "MobileNet_XGB": "#8172B2",
}
LSTM_COLOR = "#C44E52"

MAC_FIXED = {
    "FalinCNN":      22.56,
    "Falin_XGB":     22.56,
    "MobileNet_XGB": 15.56,
    "MobileNetV2":   53.72,
}

BASELINE_ALIAS = {
    "electricdevices":  "electric devices",
    "starlightcurves":  "starlightcurves",
    "twopatterns":      "two patterns",
    "facesucr":         "facesucr",
    "ecg5000":          "ecg5000",
    "forda":            "forda",
    "fordb":            "fordb",
    "yoga":             "yoga",
    "mallat":           "mallat",
    "wafer":            "wafer",
}

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_run_csv(path):
    """Returns {dataset: {model: [acc...]}} acc in [0,1]."""
    data = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(path): return data
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["Dataset"].strip()
            for m in MODEL_COLS:
                raw = row.get(m, "").strip().replace("%", "")
                if raw:
                    try: data[ds][m].append(float(raw) / 100.0)
                    except ValueError: pass
    return data

def load_lstm_acc(path):
    out = {}
    skip = ("count", "mpce")
    if not os.path.exists(path): return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds  = row.get("Dataset", "").strip()
            val = row.get("LSTM-FCN", "").strip()
            if not ds or not val: continue
            if any(ds.lower().startswith(p) for p in skip): continue
            try: out[ds.lower()] = float(val)
            except ValueError: pass
    return out

def load_lstm_macs(path):
    out = {}
    if not os.path.exists(path): return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds  = row.get("Dataset", "").strip()
            val = row.get("MACs", "").strip()
            if ds and val:
                try: out[ds.lower()] = float(val)
                except ValueError: pass
    return out

def resolve(ds_name, d):
    key = BASELINE_ALIAS.get(ds_name.lower().replace(" ",""), ds_name.lower())
    if key in d: return d[key]
    for k, v in d.items():
        if k.replace(" ","") == key.replace(" ",""): return v
    return None

# ── Load all data ─────────────────────────────────────────────────────────────
GAFS_CSV  = "results/results_gafs.csv"
STFS_CSV = "results/results_stft.csv"
LSTM_ACC  = "misc/results_LSTM.csv"
LSTM_MACS = "misc/lstm_macs.csv"

gafs_data = load_run_csv(GAFS_CSV)

# Synthetic STFS (±1.5% noise) for preview
rng = np.random.default_rng(42)
stfs_data = load_run_csv(STFS_CSV)
# for ds, models in gafs_data.items():
#     for m, runs in models.items():
#         arr = np.array(runs)
#         stfs_data[ds][m] = np.clip(arr + rng.uniform(-0.015,0.015,len(arr)),0,1).tolist()

lstm_acc  = load_lstm_acc(LSTM_ACC)
lstm_macs = load_lstm_macs(LSTM_MACS)

datasets = sorted(gafs_data.keys())
n_ds = len(datasets)

# ── Precompute per-dataset means ──────────────────────────────────────────────
def mean_acc(data, ds, model):
    runs = data.get(ds, {}).get(model, [])
    return np.mean(runs) if runs else None

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Pareto grid (2 cols × 5 rows)
# ─────────────────────────────────────────────────────────────────────────────
# Assuming a full-width figure in a standard 2-column paper (usually ~7 to 8 inches wide)
NCOLS, NROWS = 5, 2
FIG1_W = 7.5  # Reduced from 16
FIG1_H = 2.8  # Reduced from 4 to fit in 1/4 page height

fig1, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG1_W, FIG1_H),
                           gridspec_kw=dict(hspace=0.40, wspace=0.25, # Adjusted spaces for condensed look
                                            top=0.90, bottom=0.12,
                                            left=0.07, right=0.98))

# collect global acc range for shared Y
all_accs = [v for ds in datasets for m in MODEL_COLS
              for v in gafs_data[ds].get(m,[]) + stfs_data[ds].get(m,[])]
all_accs += [v for v in lstm_acc.values()]
global_ymin = max(0.0, min(all_accs) - 0.03)
global_ymax = min(1.01, max(all_accs) + 0.02)

# MACs range for X axis (fixed models + LSTM variable)
all_macs_x = list(MAC_FIXED.values()) + list(lstm_macs.values())
xmin = max(0, min(all_macs_x) * 0.7)
xmax = max(all_macs_x) * 1.15

for idx, ds in enumerate(datasets):
    row, col = divmod(idx, NCOLS)
    ax = axes[row][col]

    lstm_a = resolve(ds, lstm_acc)
    lstm_m = resolve(ds, lstm_macs)

    # -- LSTM-FCN star --
    if lstm_a is not None and lstm_m is not None:
        ax.scatter(lstm_m, lstm_a,
                   marker="*", s=60, color=LSTM_COLOR, # slightly reduced size for smaller figure
                   edgecolors="white", linewidths=0.4,
                   zorder=5, label="LSTM-FCN")

    # -- Our models: GAFS (filled ●) and STFS (open ■) --
    for m in MODEL_COLS:
        col_c = PALETTE[m]
        mac   = MAC_FIXED[m]

        # GAFS
        ga = mean_acc(gafs_data, ds, m)
        if ga is not None:
            ax.scatter(mac, ga, s=20, marker="o", color=col_c, # slightly reduced size
                       edgecolors="white", linewidths=0.4, zorder=3)

        # STFS
        sa = mean_acc(stfs_data, ds, m)
        if sa is not None:
            ax.scatter(mac, sa, s=20, marker="s", color=col_c,
                       facecolors="none", edgecolors=col_c,
                       linewidths=0.7, zorder=3)

    # --- 3. Shaded area in the top-left (ideal region) instead of an arrow ---
    # Using a triangle polygon to represent the "ideal" corner
    ideal_poly = plt.Polygon([
        (xmin, global_ymax - (global_ymax-global_ymin)*0.35),
        (xmin, global_ymax),
        (xmin + (xmax-xmin)*0.35, global_ymax)
    ], color="#2ca02c", alpha=0.15, zorder=1) # green transparent shade
    ax.add_patch(ideal_poly)

    # Optional: add a tiny text inside the shaded area
    # ax.text(xmin + (xmax-xmin)*0.02, global_ymax - (global_ymax-global_ymin)*0.05, "Ideal", fontsize=6, color="#2ca02c", alpha=0.7)

    ax.set_title(ds, pad=3) # Removed specific fontsize so it uses rcParams
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_yticks(np.linspace(global_ymin, global_ymax, 4))
    ax.set_yticklabels([f"{v*100:.0f}%" for v in np.linspace(global_ymin, global_ymax, 4)])
    
    # ── MODIFIED SECTION: Dynamic X-ticks including SOTA ──
    current_xticks = [15.56, 22.56, 53.72]
    if lstm_m is not None:
        current_xticks.append(lstm_m)
    current_xticks = sorted(list(set(current_xticks))) # Ensure order and remove duplicates if any
    
    ax.set_xticks(current_xticks)
    ax.set_xticklabels([f"{v:.1f}" for v in current_xticks], rotation=90, ha='right')
    # ──────────────────────────────────────────────────────
    
    # --- 4. Removed X-axis label as requested ---
    if row == NROWS-1 or (row == NROWS-2 and col == 1 and n_ds % 2 != 0):
        ax.set_xlabel("MACs (M)") 
    
    if col == 0:
        ax.set_ylabel("Accuracy")

    ax.grid(linestyle=":", linewidth=0.3, color="#bbbbbb")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(length=2)

# hide last empty panel if n_ds is odd
if n_ds % 2 != 0:
    axes[NROWS-1][1].set_visible(False)
# ── Shared legend for fig1 ────────────────────────────────────────────────────
leg1 = []
for m in MODEL_COLS:
    leg1.append(Line2D([0],[0], marker="o", linestyle="none",
                       color=PALETTE[m], markerfacecolor=PALETTE[m],
                       markeredgecolor="white", markeredgewidth=0.4,
                       markersize=4.5, label=f"{m} [{MAC_FIXED[m]:.1f}M MACs]"))
leg1 += [
    Line2D([0],[0], marker="*", linestyle="none", color=LSTM_COLOR,
           markersize=7, label="LSTM-FCN (Karim et al.)"),
    Line2D([0],[0], marker="o", linestyle="none", color="#555",
           markersize=4, label="GAFS  (filled)"),
    Line2D([0],[0], marker="s", linestyle="none", color="#555",
           markerfacecolor="none", markersize=4, markeredgewidth=0.7,
           label="STFS  (open)"),
]
fig1.legend(handles=leg1, loc="lower center", ncol=3,
            bbox_to_anchor=(0.50, 0.0),
            fontsize=FONT_LEGEND, frameon=True, framealpha=0.95,
            edgecolor="#cccccc", handlelength=1.2,
            columnspacing=1.0, borderpad=0.5)
fig1.suptitle("Pareto frontier: accuracy vs. computational cost per dataset",
              fontsize=FONT_TITLE, y=0.97)

fig1.savefig("results/fig1_pareto_grid.png", bbox_inches="tight")
fig1.savefig("results/fig1_pareto_grid.pdf", bbox_inches="tight")
fig1.savefig("results/fig1_pareto_grid.svg", bbox_inches="tight")

print("Fig 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Mirrored bar: best model per dataset
#   Top bar    : Δ accuracy vs LSTM-FCN  (negative = we lose acc)
#   Bottom bar : MAC savings %           (positive = we save compute)
#   GAFS and STFS side by side per dataset
# ─────────────────────────────────────────────────────────────────────────────
FIG2_W = 7.0
FIG2_H = 3.2

fig2, ax2 = plt.subplots(figsize=(FIG2_W, FIG2_H),
                          gridspec_kw=dict(top=0.88, bottom=0.22,
                                           left=0.10, right=0.97))

x = np.arange(n_ds, dtype=float)
BAR_W   = 0.30
GAP     = 0.04
OFF_G   = -(BAR_W/2 + GAP/2)   # GAFS offset
OFF_S   =  (BAR_W/2 + GAP/2)   # STFS offset

# For each dataset: find best model (highest mean acc) for GAFS and STFS
delta_acc_g, delta_acc_s   = [], []
mac_save_g,  mac_save_s    = [], []
best_model_g, best_model_s = [], []

for ds in datasets:
    lstm_a = resolve(ds, lstm_acc)
    lstm_m = resolve(ds, lstm_macs)

    # GAFS best
    best_g, best_g_acc = None, -1
    for m in MODEL_COLS:
        a = mean_acc(gafs_data, ds, m)
        if a is not None and a > best_g_acc:
            best_g_acc, best_g = a, m

    # STFS best
    best_s, best_s_acc = None, -1
    for m in MODEL_COLS:
        a = mean_acc(stfs_data, ds, m)
        if a is not None and a > best_s_acc:
            best_s_acc, best_s = a, m

    da_g = (lstm_a - best_g_acc)        if (lstm_a and best_g_acc > 0) else 0
    da_s = (lstm_a - best_s_acc)        if (lstm_a and best_s_acc > 0) else 0
    ms_g = (lstm_m - MAC_FIXED[best_g]) / lstm_m * 100 if (lstm_m and best_g) else 0
    ms_s = (lstm_m - MAC_FIXED[best_s]) / lstm_m * 100 if (lstm_m and best_s) else 0

    delta_acc_g.append(da_g)
    delta_acc_s.append(da_s)
    mac_save_g.append(ms_g)
    mac_save_s.append(ms_s)
    best_model_g.append(best_g)
    best_model_s.append(best_s)

delta_acc_g = np.array(delta_acc_g)
delta_acc_s = np.array(delta_acc_s)
mac_save_g  = np.array(mac_save_g)
mac_save_s  = np.array(mac_save_s)

# Color each bar by the best model's palette
color_g = [PALETTE[m] if m else "#888" for m in best_model_g]
color_s = [PALETTE[m] if m else "#888" for m in best_model_s]

# ── Scale: find max absolute on each side ────────────────────────────────────
acc_abs_max  = max(abs(delta_acc_g).max(), abs(delta_acc_s).max())
mac_save_max = max(mac_save_g.max(), mac_save_s.max())

# We want both axes symmetric around zero, scaled independently:
# top panel height = acc_abs_max * 1.25, bottom panel height = mac_save_max * 1.25
# But since it's one ax, we map mac savings onto negative Y:
# Y > 0: Δ acc    (actual values)
# Y < 0: −(mac_savings / mac_save_max * acc_abs_max)  → mirrored

# scaling factor so bottom bars have the same visual height as top bars
scale = acc_abs_max / mac_save_max  if mac_save_max > 0 else 1.0
mac_save_g_scaled = -mac_save_g * scale
mac_save_s_scaled = -mac_save_s * scale

# ── Draw bars ─────────────────────────────────────────────────────────────────
for i in range(n_ds):
    # GAFS acc bar (top)
    ax2.bar(x[i] + OFF_G, delta_acc_g[i], width=BAR_W,
            color=color_g[i], alpha=0.88, zorder=2)
    # STFS acc bar (top) – hatched to distinguish
    ax2.bar(x[i] + OFF_S, delta_acc_s[i], width=BAR_W,
            color=color_s[i], alpha=0.50, hatch="///",
            edgecolor=color_s[i], linewidth=0.3, zorder=2)

    # GAFS mac savings bar (bottom)
    ax2.bar(x[i] + OFF_G, mac_save_g_scaled[i], width=BAR_W,
            color=color_g[i], alpha=0.88, zorder=2)
    # STFS mac savings bar (bottom) – hatched
    ax2.bar(x[i] + OFF_S, mac_save_s_scaled[i], width=BAR_W,
            color=color_s[i], alpha=0.50, hatch="///",
            edgecolor=color_s[i], linewidth=0.3, zorder=2)

# ── Zero line ─────────────────────────────────────────────────────────────────
ax2.axhline(0, color="#555", linewidth=0.8, zorder=3)

# ── Y axis: dual label ───────────────────────────────────────────────────────
y_top = acc_abs_max * 1.3
y_bot = -mac_save_max * scale * 1.15
ax2.set_ylim(y_bot, y_top)

# top ticks in pp (percentage points)
top_ticks = np.linspace(0, acc_abs_max, 4)
all_ticks  = np.concatenate([-top_ticks[1:][::-1], top_ticks])
ax2.set_yticks(all_ticks)

def fmt_tick(v):
    if abs(v) < 1e-9: return "0"
    if v > 0:  return f"{v*100:+.1f}pp"
    # negative: convert back to MAC savings %
    return f"{(-v/scale):.0f}%"

ax2.set_yticklabels([fmt_tick(v) for v in all_ticks], fontsize=FONT_TICKS)

# Side labels
ax2.text(-0.5, acc_abs_max * 0.65, "Δ Accuracy\nvs LSTM-FCN",
         ha="right", va="center", fontsize=FONT_LEGEND, color="#333",
         linespacing=1.4)
ax2.text(-0.5, -mac_save_max * scale * 0.60, "MAC\nsavings",
         ha="right", va="center", fontsize=FONT_LEGEND, color="#333",
         linespacing=1.4)

# ── X axis ───────────────────────────────────────────────────────────────────
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, rotation=35, ha="right", fontsize=FONT_TICKS)
ax2.set_xlim(-0.6, n_ds - 0.4)

ax2.set_title("Best model per dataset: accuracy trade-off vs. MAC savings",
              fontsize=FONT_TITLE, pad=5)
ax2.grid(axis="y", linestyle=":", linewidth=0.3, color="#bbbbbb")
ax2.grid(axis="x", visible=False)
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
ax2.spines["left"].set_linewidth(0.4)
ax2.spines["bottom"].set_linewidth(0.4)

# ── Legend for fig2 ───────────────────────────────────────────────────────────
leg2 = []
for m in MODEL_COLS:
    leg2.append(mpatches.Patch(facecolor=PALETTE[m], alpha=0.88,
                               edgecolor="none", label=m))
leg2 += [
    mpatches.Patch(facecolor="#888", alpha=0.88, edgecolor="none",
                   label="Solid = GAFS"),
    mpatches.Patch(facecolor="#888", alpha=0.50, hatch="///",
                   edgecolor="#888", label="Hatched = STFS"),
]
fig2.legend(handles=leg2, loc="lower center", ncol=3,
            bbox_to_anchor=(0.50, 0.0),
            fontsize=FONT_LEGEND, frameon=True, framealpha=0.95,
            edgecolor="#cccccc", handlelength=1.2,
            columnspacing=1.0, borderpad=0.5)

fig2.savefig("results/fig2_mirror_bar.png", bbox_inches="tight")
fig2.savefig("results/fig2_mirror_bar.pdf", bbox_inches="tight")
fig2.savefig("results/fig2_mirror_bar.svg", bbox_inches="tight")

print("Fig 2 saved.")