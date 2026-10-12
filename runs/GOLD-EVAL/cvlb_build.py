"""Build the CV / gold-58 / LB calibration report (HTML with inline SVG charts) from cvlb.json."""
import html
import json

import numpy as np

T = "/Users/hllj/.claude/jobs/171aabfa/tmp/"
D = json.load(open(T + "cvlb.json"))
rows = D["rows"]
S = D["stats"]
k, c = D["fit_competitive"]
use = [r for r in rows if r["group"] != "excluded"]
fit_rows = [r for r in use if r["group"] != "OAI blend"]
GROUPS = {"single fold": "g-single", "5-fold": "g-five", "blend": "g-blend", "all-data": "g-all", "OAI blend": "g-oai"}
GLABEL = {"single fold": "Single fold", "5-fold": "5-fold ensemble", "blend": "Blend", "all-data": "All-data (Raptor v1/v2)", "OAI blend": "nartaa OAI-anchored (not in fit)"}
esc = html.escape
num = lambda x: x.replace("EXP-", "")


class Plot:
    def __init__(self, w, h, x0, x1, y0, y1, ml=58, mr=18, mt=16, mb=46):
        self.w, self.h, self.x0, self.x1, self.y0, self.y1 = w, h, x0, x1, y0, y1
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.out = []

    def X(self, v):
        return self.ml + (v - self.x0) / (self.x1 - self.x0) * (self.w - self.ml - self.mr)

    def Y(self, v):
        return self.h - self.mb - (v - self.y0) / (self.y1 - self.y0) * (self.h - self.mt - self.mb)

    def axes(self, xt, yt, xlab, ylab, fmt="{:.3f}"):
        o = self.out
        for v in yt:
            y = self.Y(v)
            o.append(f'<line class="grid" x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
            o.append(f'<text class="tick" x="{self.ml - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt.format(v)}</text>')
        for v in xt:
            x = self.X(v)
            o.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{self.mt}" y2="{self.h - self.mb}"/>')
            o.append(f'<text class="tick" x="{x:.1f}" y="{self.h - self.mb + 18}" text-anchor="middle">{fmt.format(v)}</text>')
        o.append(f'<text class="axlab" x="{(self.ml + self.w - self.mr) / 2:.1f}" y="{self.h - 8}" text-anchor="middle">{xlab}</text>')
        o.append(f'<text class="axlab" transform="translate(14 {(self.mt + self.h - self.mb) / 2:.1f}) rotate(-90)" '
                 f'text-anchor="middle">{ylab}</text>')

    def svg(self, label):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(label)}">'
                + "".join(self.out) + "</svg>")


# ---------- chart 1: gold-58 vs LB ----------
p = Plot(760, 490, 0.872, 0.934, 0.902, 0.952)
p.axes([0.875, 0.885, 0.895, 0.905, 0.915, 0.925], [0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940, 0.945, 0.950],
       "Gold-58 macro AUC", "Public LB AUC")
gx = np.array([0.895, 0.928])
rm = S["gold_lb_competitive"]["rmse"]
band = [(g, k * g + c + rm) for g in gx] + [(g, k * g + c - rm) for g in gx[::-1]]
p.out.append('<polygon class="band" points="' + " ".join(f"{p.X(a):.1f},{p.Y(b):.1f}" for a, b in band) + '"/>')
p.out.append(f'<line class="fit" x1="{p.X(gx[0]):.1f}" y1="{p.Y(k * gx[0] + c):.1f}" x2="{p.X(gx[1]):.1f}" '
             f'y2="{p.Y(k * gx[1] + c):.1f}"/>')
LAB1 = {  # exp: (dx, dy, anchor)
    "EXP-007": (8, 4, "start"), "EXP-009": (8, 4, "start"), "EXP-019": (0, 18, "middle"), "EXP-045": (-8, 4, "end"),
    "EXP-046": (0, 18, "middle"), "EXP-040": (8, 14, "start"), "EXP-056": (0, 18, "middle"), "EXP-078": (-8, 4, "end"),
    "EXP-027": (8, 14, "start"), "EXP-058": (-6, -9, "end"), "EXP-067": (8, 14, "start"), "EXP-068": (-8, -6, "end"),
    "EXP-074": (0, -11, "middle"), "EXP-071": (-9, 4, "end"), "EXP-075": (8, 14, "start"), "EXP-076": (8, 4, "start"),
    "EXP-055": (8, 14, "start"), "EXP-060": (-8, 4, "end"),
    "EXP-077": (-8, 4, "end"), "EXP-083": (0, 16, "middle"), "EXP-086": (8, 4, "start"), "EXP-069": (8, 4, "start"),
}
for r in sorted(use, key=lambda r: r["group"]):
    if r["gold"] is None:
        continue
    x, y = p.X(r["gold"]), p.Y(r["lb"])
    tip = f'{r["exp"]} {r["label"]}: gold {r["gold"]:.4f}, LB {r["lb"]:.3f}'
    p.out.append(f'<circle class="pt {GROUPS[r["group"]]}" cx="{x:.1f}" cy="{y:.1f}" r="5.5"><title>{esc(tip)}</title></circle>')
    if r["exp"] in LAB1:
        dx, dy, a = LAB1[r["exp"]]
        p.out.append(f'<text class="plab" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{a}">{num(r["exp"])}</text>')
chart1 = p.svg("Scatter of gold-58 AUC against public LB AUC for every scored experiment")

# ---------- chart 2: residual vs the gold fit ----------
res = sorted([r for r in fit_rows if r["resid"] is not None], key=lambda r: r["resid"])
bh, top = 19, 10
H = top + bh * len(res) + 40
p2 = Plot(760, H, -0.0108, 0.0055, 0, 1, ml=330, mr=24, mt=top, mb=40)
for v in [-0.010, -0.008, -0.006, -0.004, -0.002, 0, 0.002, 0.004]:
    x = p2.X(v)
    p2.out.append(f'<line class="{"zero" if v == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{H - 40}"/>')
    p2.out.append(f'<text class="tick" x="{x:.1f}" y="{H - 22}" text-anchor="middle">{v:+.3f}</text>')
p2.out.append(f'<text class="axlab" x="{(330 + 736) / 2:.0f}" y="{H - 4}" text-anchor="middle">'
              f'LB − LB predicted from gold-58 (fit on the competitive range)</text>')
for i, r in enumerate(res):
    y = top + i * bh
    x0, x1 = p2.X(0), p2.X(r["resid"])
    p2.out.append(f'<rect class="bar {GROUPS[r["group"]]}" x="{min(x0, x1):.1f}" y="{y + 3}" width="{abs(x1 - x0):.1f}" '
                  f'height="{bh - 6}"><title>{esc(r["exp"])}: LB {r["lb"]:.3f}, predicted {r["pred_from_gold"]:.4f}</title></rect>')
    p2.out.append(f'<text class="rlab" x="322" y="{y + bh - 5}" text-anchor="end">{esc(num(r["exp"]))} · {esc(r["label"])}</text>')
    tx = x1 + (6 if r["resid"] >= 0 else -6)
    p2.out.append(f'<text class="rval" x="{tx:.1f}" y="{y + bh - 5}" text-anchor="{"start" if r["resid"] >= 0 else "end"}">'
                  f'{r["resid"]:+.4f}</text>')
chart2 = p2.svg("Bar chart of how far each experiment's LB lies above or below the gold-58 prediction")

# ---------- chart 3: fold OOF vs LB ----------
p3 = Plot(760, 430, 0.860, 0.892, 0.902, 0.938)
p3.axes([0.860, 0.865, 0.870, 0.875, 0.880, 0.885, 0.890], [0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935],
        "Single-fold OOF macro AUC (report labels)", "Public LB AUC")
f0 = [r for r in use if r["fold"] == 0 and r["oof"] is not None]
f1 = [r for r in use if r["fold"] == 1 and r["oof"] is not None]
kk, cc = np.polyfit([r["oof"] for r in f0], [r["lb"] for r in f0], 1)
xs = np.array([0.862, min(0.890, (0.937 - cc) / kk)])
p3.out.append(f'<line class="fit" x1="{p3.X(xs[0]):.1f}" y1="{p3.Y(kk * xs[0] + cc):.1f}" x2="{p3.X(xs[1]):.1f}" '
              f'y2="{p3.Y(kk * xs[1] + cc):.1f}"/>')
LAB3 = {"EXP-007": (8, 4, "start"), "EXP-019": (-8, -6, "end"), "EXP-025": (8, 16, "start"), "EXP-040": (-8, 4, "end"),
        "EXP-045": (-8, -6, "end"), "EXP-046": (0, 18, "middle"), "EXP-047": (-8, -7, "end"), "EXP-056": (8, 14, "start"),
        "EXP-078": (8, 4, "start"), "EXP-008": (8, 4, "start"), "EXP-021": (8, 4, "start"), "EXP-029": (-2, -10, "end")}
for r in f0 + f1:
    x, y = p3.X(r["oof"]), p3.Y(r["lb"])
    cls = "pt g-f0" if r["fold"] == 0 else "pt g-f1"
    tip = f'{r["exp"]} {r["label"]} (fold {r["fold"]}): OOF {r["oof"]:.4f}, LB {r["lb"]:.3f}'
    p3.out.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="5.5"><title>{esc(tip)}</title></circle>')
    if r["exp"] in LAB3:
        dx, dy, a = LAB3[r["exp"]]
        p3.out.append(f'<text class="plab" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{a}">{num(r["exp"])}</text>')
for a, b in [("EXP-007", "EXP-008"), ("EXP-019", "EXP-021"), ("EXP-025", "EXP-029")]:
    ra, rb = next(r for r in f0 if r["exp"] == a), next(r for r in f1 if r["exp"] == b)
    p3.out.append(f'<line class="pair" x1="{p3.X(ra["oof"]):.1f}" y1="{p3.Y(ra["lb"]):.1f}" '
                  f'x2="{p3.X(rb["oof"]):.1f}" y2="{p3.Y(rb["lb"]):.1f}"/>')
chart3 = p3.svg("Scatter of single-fold OOF AUC against LB for fold 0 and fold 1 models")

# ---------- chart 4: LB over time ----------
order = [r for r in use]
p4 = Plot(760, 300, -0.5, len(order) - 0.5, 0.900, 0.952, ml=58, mb=58)
for v in [0.905, 0.915, 0.925, 0.935, 0.945, 0.950]:
    y = p4.Y(v)
    p4.out.append(f'<line class="grid" x1="{p4.ml}" x2="{p4.w - p4.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
    p4.out.append(f'<text class="tick" x="{p4.ml - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.3f}</text>')
best, path = 0, []
for i, r in enumerate(order):
    best = max(best, r["lb"])
    path.append((p4.X(i - 0.5), p4.Y(best)))
    path.append((p4.X(i + 0.5), p4.Y(best)))
p4.out.append('<polyline class="best" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in path) + '"/>')
for i, r in enumerate(order):
    x, y = p4.X(i), p4.Y(r["lb"])
    p4.out.append(f'<circle class="pt {GROUPS[r["group"]]}" cx="{x:.1f}" cy="{y:.1f}" r="4.5">'
                  f'<title>{esc(r["exp"])} {esc(r["label"])}: LB {r["lb"]:.3f}</title></circle>')
    p4.out.append(f'<text class="xlab" transform="translate({x + 3:.1f} {p4.h - p4.mb + 10}) rotate(60)">{num(r["exp"])}</text>')
p4.out.append(f'<text class="axlab" transform="translate(14 {(p4.mt + p4.h - p4.mb) / 2:.1f}) rotate(-90)" '
              f'text-anchor="middle">Public LB AUC</text>')
chart4 = p4.svg("Public LB of every scored experiment in submission order, with the best score so far")

# ---------- table ----------
trs = []
for r in rows:
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"
    cls = ' class="muted-row"' if r["group"] == "excluded" else ""
    resid_txt = "—" if r["resid"] is None else f'{r["resid"]:+.4f}'
    trs.append(f'<tr{cls}><td class="mono">{num(r["exp"])}</td><td>{esc(r["label"])}</td>'
               f'<td><span class="chip {GROUPS.get(r["group"], "g-ex")}"></span>{esc(r["group"])}</td>'
               f'<td class="mono r">{f(r["oof"])}</td><td class="mono r">{f(r["gold"])}</td>'
               f'<td class="mono r">{f(r["lb"], 3)}</td><td class="mono r">{resid_txt}</td></tr>')
table = "\n".join(trs)
legend = "".join(f'<span class="leg"><span class="chip {v}"></span>{GLABEL[k]}</span>' for k, v in GROUPS.items())

g_all, g_cmp, o0, oall, go = (S[k] for k in ["gold_lb_all", "gold_lb_competitive", "oof_lb_fold0",
                                             "oof_lb_all_with_oof", "gold_oof"])
page = open(T + "cvlb_template.html").read()
for key, val in {
    "{{CHART1}}": chart1, "{{CHART2}}": chart2, "{{CHART3}}": chart3, "{{CHART4}}": chart4, "{{TABLE}}": table,
    "{{LEGEND}}": legend, "{{N_ALL}}": str(g_all["n"]), "{{G_P}}": f'{g_cmp["pearson"]:.2f}',
    "{{G_S}}": f'{g_cmp["spearman"]:.2f}', "{{G_N}}": str(g_cmp["n"]), "{{G_RMSE}}": f'{g_cmp["rmse"]:.4f}',
    "{{G_SLOPE}}": f'{k:.2f}', "{{G_ICPT}}": f'{c:.3f}', "{{GA_P}}": f'{g_all["pearson"]:.2f}',
    "{{O0_P}}": f'{o0["pearson"]:.2f}', "{{O0_N}}": str(o0["n"]), "{{O0_SLOPE}}": f'{kk:.2f}',
    "{{OA_P}}": f'{oall["pearson"]:.2f}', "{{OA_N}}": str(oall["n"]), "{{GO_P}}": f'{go["pearson"]:.2f}',
    "{{GO_N}}": str(go["n"]), "{{OFF_M}}": f'{S["lb_minus_gold_mean"]:+.4f}', "{{OFF_SD}}": f'{S["lb_minus_gold_sd"]:.4f}',
}.items():
    page = page.replace(key, val)
assert "{{" not in page, page[page.index("{{"):page.index("{{") + 40]
open(T + "cvlb_report.html", "w").write(page)
print("written", len(page))
