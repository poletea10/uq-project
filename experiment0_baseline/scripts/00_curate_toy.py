"""Toy-function curation (phase0_protocol.md §2.2). Model-free: it looks at f only, never at a model output.

For toy_seed = 0, 1, ..., draws (theta, eta) and keeps the first seed whose f meets all three pre-registered
criteria of toy_function.yaml `curation`, printing the statistics of every candidate tried.
- toy_seed still null: writes the chosen seed into toy_function.yaml; it is frozen from then on.
- toy_seed already set: re-checks that seed and changes nothing.
Writes outputs/figures/toy/00_toy_function.png (f, f_base, f_gap, and each toy rung's n_max context).
Non-zero exit if no candidate passes, or if a frozen seed fails the criteria.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import config
import figures
import toy


def fmt(stats):
    return (f"max|f| {stats['max_abs_f']:.3f}  sd(f_gap) {stats['gap_sd']:.3f}  "
            f"range(f) on noise strip {stats['noise_strip_range']:.3f}")


def main():
    proto, tcfg = config.protocol(), config.toy_function()
    base = proto["base_seed"]
    print(f"s_theta {toy.theta_scale(tcfg):.4f}, s_eta {toy.eta_scale(tcfg):.4f}")
    print(f"criteria: {tcfg['curation']}")

    frozen = tcfg["toy_seed"]
    if frozen is not None:
        stats = toy.curation_stats(toy.draw_function(frozen, tcfg, base))
        ok = toy.passes_curation(stats, tcfg)
        print(f"toy_seed {frozen} is frozen: {fmt(stats)}  -> {'passes' if ok else 'FAILS'}")
        seed = frozen
    else:
        seed = None
        for s in range(tcfg["curation"]["max_seed"]):
            stats = toy.curation_stats(toy.draw_function(s, tcfg, base))
            ok = toy.passes_curation(stats, tcfg)
            print(f"seed {s:4d}: {fmt(stats)}  {'PASS' if ok else 'reject'}", flush=True)
            if ok:
                seed = s
                break
        if seed is None:
            print(f"no seed in [0, {tcfg['curation']['max_seed']}) passes")
            sys.exit(1)
        config.write_toy_seed(seed)
        tcfg = config.toy_function()
        print(f"toy_seed {seed} written to {config.CONFIG_DIR / 'toy_function.yaml'}")

    tasks = {k: toy.load_task(k, proto, tcfg) for k in config.TOY_RUNGS}
    t0 = tasks[0]
    a, b = tcfg["sparse_strip"]
    in_gap = (t0.x_ctx >= a) & (t0.x_ctx <= b)
    for n in proto["n_grid"]:
        print(f"     n={n:5d}: {int(in_gap[:n].sum())} context points in the sparse strip")
    for k, task in tasks.items():
        s = task.sigma(task.x_qry)
        print(f"     toy rung {k}: sigma over queries in [{s.min():.3f}, {s.max():.3f}]")
    out = config.FIG_DIR / "toy" / "00_toy_function.png"
    figures.toy_function(tasks, toy.curation_stats(t0.fn), proto["n_grid"], out)
    print(f"figure: {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
