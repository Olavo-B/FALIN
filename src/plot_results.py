
import os, re, csv, warnings
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# =============================================================================
# A.  FONT / STYLE
# =============================================================================
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

# =============================================================================
# B.  PATHS
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
CSV_GAFS     = os.path.join(BASE_DIR, "results", "results_gafs.csv")
CSV_STFS     = os.path.join(BASE_DIR, "results", "results_stfs.csv")
CSV_BASELINE = os.path.join(BASE_DIR, "misc", "results_LSTM.csv")
CSV_LSTM_MACS= os.path.join(BASE_DIR, "mock_lstm_macs.csv")  # <-- NOVO ARQUIVO DE MOCK
OUTPUT_PNG   = os.path.join(RESULTS_DIR, "dumbbell_gafs_stfs.png")
OUTPUT_PDF   = os.path.join(RESULTS_DIR, "dumbbell_gafs_stfs.pdf")

# =============================================================================
# C.  DESIGN CONSTANTS
# =============================================================================
MODEL_COLS = ["FalinCNN", "Falin_XGB", "MobileNetV2", "MobileNet_XGB"]

PALETTE = {
    "FalinCNN":      "#4C72B0",
    "Falin_XGB":     "#55A868",
    "MobileNetV2":   "#DD8452",
    "MobileNet_XGB": "#8172B2",
}
LSTM_COLOR = "#C44E52"

BASELINE_ALIAS = {
    "electricdevices": "electric devices",
    "starlightcurves": "starlightcurves",
    "twopatterns":     "two patterns",
    "facesucr":        "facesucr",
    "ecg5000":         "ecg5000",
    "forda":           "forda",
    "fordb":           "fordb",
    "yoga":            "yoga",
    "mallat":          "mallat",
    "wafer":           "wafer",
}

# =============================================================================
# D.  DATA LOADING
# =============================================================================
def load_run_csv(path: str) -> dict:
    data = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(path):
        return data
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row["Dataset"].strip()
            for m in MODEL_COLS:
                raw = row.get(m, "").strip().replace("%", "")
                if raw:
                    try:
                        data[ds][m].append(float(raw) / 100.0)
                    except ValueError:
                        pass
    return data

def load_baselines(path: str, column_name: str = "LSTM-FCN") -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds  = row.get("Dataset", "").strip()
            val = row.get(column_name, "").strip()
            if ds and val:
                try:
                    out[ds.lower()] = float(val)
                except ValueError:
                    pass
    return out

def resolve_baseline(ds_name: str, baselines: dict) -> float | None:
    key = BASELINE_ALIAS.get(ds_name.lower().replace(" ", ""), ds_name.lower())
    if key in baselines:
        return baselines[key]
    key2 = key.replace(" ", "")
    for k, v in baselines.items():
        if k.replace(" ", "") == key2:
            return v
    return None

def parse_mac_file(path: str) -> dict:
    macs = {}
    if not os.path.exists(path):
        return macs
    pattern = re.compile(r"^\s*([\w\-]+)\s*\|\s*([\d.]+)\s*M", re.IGNORECASE)
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                name, val = m.group(1), m.group(2)
                if name.lower() in ("model", "layer"):
                    continue
                macs[name] = float(val)
    return macs

def gather_macs(results_dir: str) -> dict:
    # Gathering non-LSTM MACs
    for aug in ("gafs", "stfs"):
        aug_path = os.path.join(results_dir, aug)
        if not os.path.isdir(aug_path):
            continue
        for ds_folder in sorted(os.listdir(aug_path)):
            ds_path = os.path.join(aug_path, ds_folder)
            if not os.path.isdir(ds_path):
                continue
            for run_folder in sorted(os.listdir(ds_path)):
                mac_file = os.path.join(ds_path, run_folder,
                                        "hardware_complexity_MACs.txt")
                parsed = parse_mac_file(mac_file)
                if parsed:
                    return parsed
    return {}

# =============================================================================
# E.  PANEL DRAWING
# =============================================================================
def draw_panel(ax, data, datasets, baselines, x_centers, jitter, title, show_xlabel):
    n_ds  = len(datasets)
    
    for xi, ds in enumerate(datasets):
        lstm = resolve_baseline(ds, baselines)
        if lstm is not None:
            ax.plot([x_centers[xi] - 0.3, x_centers[xi] + 0.3],
                    [lstm, lstm],
                    color=LSTM_COLOR, linewidth=1.2,
                    linestyle="--", zorder=1, alpha=0.9,
                    solid_capstyle="butt")

    all_acc = []
    for xi, ds in enumerate(datasets):
        for j, model in enumerate(MODEL_COLS):
            runs = data.get(ds, {}).get(model, [])
            if not runs:
                continue
            arr  = np.array(runs)
            mean = arr.mean()
            lo   = arr.min()
            hi   = arr.max()
            col  = PALETTE[model]
            
            xp = x_centers[xi] + jitter[j]
            all_acc.extend(arr.tolist())

            ax.vlines(xp, lo, hi, color=col, linewidth=1.2, zorder=2, alpha=0.75)
            ax.scatter(xp, mean, color=col, s=22, marker="o",
                       edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlim(x_centers[0] - 0.6, x_centers[-1] + 0.6)
    ax.set_xticks(x_centers)

    if show_xlabel:
        ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=FONT_TICKS)
    else:
        ax.set_xticklabels([])

    if all_acc:
        ymin = max(0.0, min(all_acc) - 0.04)
        ymax = min(1.0, max(all_acc) + 0.04)
    else:
        ymin, ymax = 0.5, 1.0

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.set_yticklabels([f"{v*100:.0f}%" for v in np.linspace(ymin, ymax, 5)], fontsize=FONT_TICKS)
    ax.set_ylabel("Accuracy", fontsize=FONT_AXES)
    ax.set_title(title, fontsize=FONT_TITLE, pad=5)

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#bbbbbb")

    for xi in range(n_ds - 1):
        ax.axvline(x_centers[xi] + 0.5, color="#e0e0e0", linewidth=0.5, zorder=0)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


def draw_mac_panel(ax, mac_table, lstm_macs, datasets, x_centers):
    plot_models = MODEL_COLS + ["LSTM-FCN"]
    mac_colors = {**PALETTE, "LSTM-FCN": LSTM_COLOR}
    
    n_mdl = len(plot_models)
    n_ds = len(datasets)
    
    total_width = 0.75
    bar_w = total_width / n_mdl
    offsets = np.linspace(-total_width/2 + bar_w/2, total_width/2 - bar_w/2, n_mdl)
    
    max_mac = 0
    for xi, ds in enumerate(datasets):
        for j, model in enumerate(plot_models):
            
            if model == "LSTM-FCN":
                mac_val = resolve_baseline(ds, lstm_macs)
                if mac_val is None:
                    mac_val = 0.0
            else:
                mac_val = mac_table.get(model, 0)
                if mac_val == 0:
                    for k, v in mac_table.items():
                        if k.lower() == model.lower() or k.replace("-", "").lower() == model.replace("-", "").lower():
                            mac_val = v
                            break
                        
            if mac_val > max_mac: 
                max_mac = mac_val
            
            ax.bar(x_centers[xi] + offsets[j], mac_val, width=bar_w*0.85, 
                   color=mac_colors[model], alpha=0.85, zorder=3)

    ax.set_xlim(x_centers[0] - 0.6, x_centers[-1] + 0.6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=FONT_TICKS)
    ax.set_xlabel("Dataset", fontsize=FONT_AXES)
    
    ax.set_ylabel("MACs (M)", fontsize=FONT_AXES)
    ax.set_title("Hardware Complexity (MACs)", fontsize=FONT_TITLE, pad=5)
    
    if max_mac > 0:
        ax.set_ylim(0, max_mac * 1.15)
        
    ax.grid(axis="y", linestyle=":", linewidth=0.3, color="#bbbbbb")
    ax.grid(axis="x", visible=False)
    
    for xi in range(n_ds - 1):
        ax.axvline(x_centers[xi] + 0.5, color="#e0e0e0", linewidth=0.5, zorder=0)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# =============================================================================
# F.  MAIN
# =============================================================================
def main():
    gafs_data  = load_run_csv(CSV_GAFS)
    stfs_data  = load_run_csv(CSV_STFS)
    baselines  = load_baselines(CSV_BASELINE, column_name="LSTM-FCN")
    mac_table  = gather_macs(RESULTS_DIR)
    
    # Load dataset-specific MACs for LSTM
    lstm_macs  = load_baselines(CSV_LSTM_MACS, column_name="MACs")

    datasets = sorted(set(gafs_data.keys()) | set(stfs_data.keys()))
    if not datasets:
        print("[WARN] No datasets found. Provide the CSVs to test fully.")
        # Fallback to test
        datasets = ["ecg5000", "forda", "electricdevices", "yoga", "mallat"]
        
    n_ds = len(datasets)
    x_centers = np.arange(n_ds, dtype=float)

    jitter = np.linspace(-0.06, 0.06, len(MODEL_COLS))

    FIG_W = 7.0
    FIG_H = max(6.5, n_ds * 0.6)

    fig, (ax_g, ax_s, ax_m) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(height_ratios=[1, 1, 0.7],
                         hspace=0.45,
                         top=0.95, bottom=0.15,
                         left=0.10, right=0.95)
    )

    draw_panel(ax_g, gafs_data, datasets, baselines, x_centers, jitter,
               title="GAFS", show_xlabel=False)

    draw_panel(ax_s, stfs_data, datasets, baselines, x_centers, jitter,
               title="STFS", show_xlabel=False)
               
    draw_mac_panel(ax_m, mac_table, lstm_macs, datasets, x_centers)

    leg = []
    for model in MODEL_COLS:
        leg.append(Line2D([0], [0], marker="o", color=PALETTE[model],
            markerfacecolor=PALETTE[model], markeredgecolor="white",
            markeredgewidth=0.5, markersize=5, linestyle="-", linewidth=1.2,
            label=f"{model}"))
    
    leg.append(mpatches.Patch(color=LSTM_COLOR, label="LSTM-FCN (MACs)"))
    leg.append(Line2D([0], [0], color=LSTM_COLOR, linewidth=1.0, linestyle="--", label="LSTM-FCN (Baseline Acc)"))

    fig.legend(
        handles=leg,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=FONT_LEGEND,
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=1.6,
        columnspacing=1.2,
        borderpad=0.6,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"[OK] Saved!")

if __name__ == "__main__":
    main()
