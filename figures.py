import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
S = json.load(open("summary.json"))
OUT = "figures/"

LABEL = {"monolithic": "Monolithic", "acl_contexts": "Authorized contexts", "ddd_oracle": "Oracle routing",
         "ddd_learned": "Learned router", "ddd_routed": "TF-IDF router", "ddd_random": "Random routing",
         "leiden_top1": "Leiden top-1", "leiden_top3": "Leiden top-3", "leiden_top3_acl": "Leiden top-3 + ACL",
         "ddd_oracle_no_identity": "Oracle, no identity"}

# Fig 2: accuracy vs nodes explored at the largest size
T = S["table"]["8000"]
fig, ax = plt.subplots(figsize=(3.45, 2.35))
for m, lab in LABEL.items():
    d = T[m]
    leaks = d["leak"][0] > 0.5
    ax.errorbar(d["explored"][0], d["hit1"][0], yerr=d["hit1"][1], fmt="o" if leaks else "s",
                ms=4.5, mfc="#c0392b" if leaks else "white", mec="#c0392b" if leaks else "#1f4e79",
                ecolor="#888", elinewidth=0.6, capsize=1.5, zorder=3)
    off = {"monolithic": (0, 7), "acl_contexts": (-5, -12), "ddd_oracle": (-5, 4), "ddd_learned": (5, -8),
           "ddd_routed": (0, -12), "ddd_random": (5, -3), "leiden_top1": (5, -5), "leiden_top3": (5, 4),
           "leiden_top3_acl": (-5, 4), "ddd_oracle_no_identity": (5, 2)}[m]
    ax.annotate(lab, (d["explored"][0], d["hit1"][0]), textcoords="offset points", xytext=off, fontsize=6.3,
                ha="right" if off[0] < 0 else ("center" if off[0] == 0 else "left"))
ax.set_xlabel("Nodes explored per query (mean)")
ax.set_ylabel("Hit@1 (correct entity)")
ax.set_xlim(0, 205); ax.set_ylim(0.1, 0.55)
ax.grid(alpha=0.25, lw=0.4)
ax.scatter([], [], marker="o", c="#c0392b", s=14, label="touches unauthorized contexts")
ax.scatter([], [], marker="s", facecolors="white", edgecolors="#1f4e79", s=14, label="stays within authorization")
ax.legend(loc="upper left", fontsize=6.3, frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig_tradeoff.pdf")
fig.savefig(OUT + "fig_tradeoff.png", dpi=200)

# Fig 1: architecture schematic
fig, ax = plt.subplots(figsize=(3.45, 2.1))
ax.set_xlim(-1, 101); ax.set_ylim(0, 64); ax.axis("off")
def box(x, y, w, h, txt, fc="white", ec="#1f4e79", fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#444"))
box(1, 50, 24, 11, "Query + role", fc="#f4f4f4", ec="#555", fs=6)
box(35, 50, 26, 11, "Policy: authorized\ncontexts of role", fc="#f4f4f4", ec="#555", fs=6)
box(71, 50, 28, 11, "Router narrows\nthe contexts", fc="#f4f4f4", ec="#555", fs=6)
arrow(26, 55.5, 34, 55.5); arrow(62, 55.5, 70, 55.5)
ctxs = [("CUSTOMER", 2), ("PAYMENTS", 21.5), ("LENDING", 41), ("RISK", 60.5), ("SUPPORT", 80)]
for name, x in ctxs:
    sel = name in ("CUSTOMER", "PAYMENTS", "RISK")
    box(x, 22, 17.5, 14, name.title(), fc="#e8f0f8" if sel else "white", ec="#1f4e79" if sel else "#aaa", fs=5.8)
box(1, 3, 98, 10, "Shared identity kernel (Party nodes)", fc="#fbf3e4", ec="#b9770e", fs=6)
for _, x in ctxs:
    ax.plot([x + 8.75, x + 8.75], [22, 13], color="#b9770e", lw=0.6, ls=":")
arrow(85, 50, 85, 37.5)
ax.text(50, 43.5, "scope = selected contexts + identity kernel", fontsize=5.8, ha="center", va="center", style="italic")
fig.tight_layout(pad=0.1)
fig.savefig(OUT + "fig_arch.pdf")
fig.savefig(OUT + "fig_arch.png", dpi=200)
print("ok")
