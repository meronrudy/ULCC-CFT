
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

# Backend selection: prefer TkAgg when Tk is available; fallback to Agg.
def _tk_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False

if _tk_available():
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

# Canvas import only when Tk is usable; guard at runtime otherwise.
if _tk_available():
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
else:
    FigureCanvasTkAgg = None  # type: ignore

from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Project imports (slow-plane + harness)
from slow_plane.perf_overhead import _synthetic_pggs_artifacts as syn_artifacts  # type: ignore[attr-defined]
from slow_plane.field.solver import FieldConfig, solve_field
from slow_plane.geometry.update import GeometryConfig, update_geometry
from harness.control_loop import control_apply_cycle
from harness.e4_morphogenesis_bench import (
    run_morphogenesis_sweep,
    write_morphogenesis_report,
    _fast_plane_probe,  # internal probe for KPIs
)


@dataclass
class LiveState:
    H: int
    W: int
    g: List[List[List[List[float]]]]  # [H][W][2][2]
    U: List[List[float]]
    J: List[List[float]]
    B: Dict[str, List[List[float]]]
    geom_cfg: GeometryConfig


def _mk_identity_metric(H: int, W: int) -> List[List[List[List[float]]]]:
    row = [[[1.0, 0.0], [0.0, 1.0]] for _ in range(W)]
    return [list(row) for _ in range(H)]


def _percentile_nearest_rank(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    p = max(0.0, min(100.0, float(p)))
    k = int(round((p / 100.0) * (len(v) - 1)))
    return float(v[k])


def _geom_cfg_from_label(label: str) -> Optional[GeometryConfig]:
    table = {
        "baseline": None,
        "adapt_tr0.10_hyst0": GeometryConfig(trust_radius=0.10, hysteresis=0),
        "adapt_tr0.25_hyst2": GeometryConfig(trust_radius=0.25, hysteresis=2),
        "adapt_tr0.50_hyst4": GeometryConfig(trust_radius=0.50, hysteresis=4),
        "adapt_tr1.00_hyst2": GeometryConfig(trust_radius=1.00, hysteresis=2),
    }
    return table.get(label, None)


class MorphogenesisGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ULCC-CFT Morphogenesis Bench")
        self.geometry("1200x800")
        self._running = False
        self._worker: Optional[threading.Thread] = None

        # Parameters (Tk variables)
        self.var_grid = tk.IntVar(value=8)
        self.var_iters = tk.IntVar(value=50)
        self.var_fp_cycles = tk.IntVar(value=2000)
        self.var_period_us = tk.IntVar(value=1_000_000)
        self.var_geom = tk.StringVar(value="baseline")
        self.var_status = tk.StringVar(value="Idle")
        # Custom geometry sliders
        self.var_trust = tk.DoubleVar(value=0.25)
        self.var_hyst = tk.IntVar(value=2)
        # Simple state persistence path
        self._state_path = os.path.join(os.path.dirname(__file__), ".gui_state.json")

        # Build UI
        self._build_controls()
        self._build_plots()
        self._build_metrics_table()
        # Load persisted state if any
        self._load_state()
        # Save on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Live buffers
        self._live_times: List[float] = []
        self._live_accept: List[float] = []
        # Comparison summaries
        self._summary_a: Optional[Dict[str, Any]] = None
        self._summary_b: Optional[Dict[str, Any]] = None
        self._live_resid: List[float] = []

    # ---------------- UI building ----------------

    def _build_controls(self) -> None:
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Grid size
        ttk.Label(frm, text="Grid N:").pack(side=tk.LEFT)
        cb_grid = ttk.Combobox(frm, textvariable=self.var_grid, width=5, values=(4, 8, 16, 32), state="readonly")
        cb_grid.pack(side=tk.LEFT, padx=6)

        # Iterations
        ttk.Label(frm, text="Iterations:").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Entry(frm, textvariable=self.var_iters, width=8).pack(side=tk.LEFT, padx=6)

        # FP cycles
        ttk.Label(frm, text="FP cycles:").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Entry(frm, textvariable=self.var_fp_cycles, width=8).pack(side=tk.LEFT, padx=6)

        # Period (us)
        ttk.Label(frm, text="Period (us):").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Entry(frm, textvariable=self.var_period_us, width=10).pack(side=tk.LEFT, padx=6)

        # Geometry config
        ttk.Label(frm, text="Geometry:").pack(side=tk.LEFT, padx=(16, 0))
        cb_geom = ttk.Combobox(
            frm,
            textvariable=self.var_geom,
            width=18,
            values=("baseline", "adapt_tr0.10_hyst0", "adapt_tr0.25_hyst2", "adapt_tr0.50_hyst4", "adapt_tr1.00_hyst2", "custom"),
            state="readonly",
        )
        cb_geom.pack(side=tk.LEFT, padx=6)

        # Custom geometry sliders (shown regardless; used when mode=custom)
        ttk.Label(frm, text="trust_radius").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Scale(frm, from_=0.01, to=2.0, orient=tk.HORIZONTAL, variable=self.var_trust, length=120).pack(side=tk.LEFT, padx=6)
        ttk.Label(frm, text="hysteresis").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Scale(frm, from_=0, to=8, orient=tk.HORIZONTAL, variable=self.var_hyst, length=100).pack(side=tk.LEFT, padx=6)

        # Buttons
        ttk.Button(frm, text="Live Morphogenesis", command=self._on_live).pack(side=tk.LEFT, padx=(24, 6))
        ttk.Button(frm, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Run Sweep + Report", command=self._on_sweep_report).pack(side=tk.LEFT, padx=12)
        ttk.Button(frm, text="Export Latest Plot", command=self._on_export_plot).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Export Live Data", command=self._on_export_data).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="FP Probe", command=self._on_fp_probe).pack(side=tk.LEFT, padx=6)
        # Load/compare benchmark summaries
        ttk.Button(frm, text="Load Summary A", command=lambda: self._on_load_summary('A')).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Load Summary B", command=lambda: self._on_load_summary('B')).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Compare Summaries", command=self._on_compare_summaries).pack(side=tk.LEFT, padx=6)
        # Presets
        ttk.Button(frm, text="Preset: Quick", command=lambda: self._apply_preset("quick")).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Preset: Extended", command=lambda: self._apply_preset("extended")).pack(side=tk.LEFT, padx=6)

        ttk.Label(frm, textvariable=self.var_status).pack(side=tk.RIGHT)

    def _build_plots(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: Heatmap (Phi)
        frm_left = ttk.Labelframe(paned, text="Morphogenetic Field (Φ)")
        paned.add(frm_left, weight=1)
        self.fig_left = Figure(figsize=(5, 4), dpi=100)
        self.ax_phi = self.fig_left.add_subplot(111)
        self.im_phi = None
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=frm_left)
        self.canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: Metrics chart (timing + acceptance)
        frm_right = ttk.Labelframe(paned, text="Timing/Acceptance")
        paned.add(frm_right, weight=1)
        self.fig_right = Figure(figsize=(5, 4), dpi=100)
        self.ax_time = self.fig_right.add_subplot(211)
        self.ax_acc = self.fig_right.add_subplot(212)
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=frm_right)
        self.canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_time.set_title("Control Apply Time (us)")
        self.ax_time.set_ylabel("us")
        self.ax_acc.set_title("Trust-region Acceptance Ratio")
        self.ax_acc.set_ylim(0, 1.0)
        # Residual overlay axis
        self.ax_acc2 = self.ax_acc.twinx()
        self.ax_acc2.set_ylabel("residual_norm", color="tab:orange")
        self.ax_acc2.tick_params(axis="y", labelcolor="tab:orange")
        self.fig_right.tight_layout()

    def _build_metrics_table(self) -> None:
        frm = ttk.Labelframe(self, text="Summary Metrics")
        frm.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        cols = ("min", "p50", "mean", "p95", "max", "accepted_ratio")
        self.tbl = ttk.Treeview(frm, columns=cols, show="headings", height=1)
        for c in cols:
            self.tbl.heading(c, text=c)
            self.tbl.column(c, width=110, anchor="center")
        self.tbl.pack(side=tk.LEFT, padx=4, pady=4)
        self._update_summary_table([], 0.0)

    def _update_summary_table(self, times_us: List[float], acc_ratio: float) -> None:
        # Compute nearest-rank percentiles
        if not times_us:
            stats = {"min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        else:
            stats = {
                "min": float(min(times_us)),
                "p50": float(_percentile_nearest_rank(times_us, 50.0)),
                "mean": float(sum(times_us) / len(times_us)),
                "p95": float(_percentile_nearest_rank(times_us, 95.0)),
                "max": float(max(times_us)),
            }
        # Clear and insert single row
        for item in self.tbl.get_children():
            self.tbl.delete(item)
        self.tbl.insert(
            "",
            "end",
            values=(
                f"{stats['min']:.1f}",
                f"{stats['p50']:.1f}",
                f"{stats['mean']:.1f}",
                f"{stats['p95']:.1f}",
                f"{stats['max']:.1f}",
                f"{acc_ratio:.3f}",
            ),
        )

    # ---------------- Actions ----------------

    def _on_live(self) -> None:
        if self._running:
            return
        try:
            N = int(self.var_grid.get())
            iters = int(self.var_iters.get())
        except Exception:
            messagebox.showerror("Invalid input", "Grid/iterations must be integers.")
            return

        self._running = True
        self.var_status.set("Running live morphogenesis...")
        self._live_times.clear()
        self._live_accept.clear()
        self._live_resid.clear()

        # Start worker thread
        self._worker = threading.Thread(target=self._live_worker, args=(N, iters), daemon=True)
        self._worker.start()

    def _on_stop(self) -> None:
        self._running = False
        self.var_status.set("Stopped")

    def _on_export_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Timing Plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            self.fig_right.savefig(path, dpi=150)
            messagebox.showinfo("Export", f"Saved plot to {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _on_export_data(self) -> None:
        # Export CSV of live series and JSON snapshot of parameters
        if not self._live_times:
            messagebox.showwarning("No data", "Run live morphogenesis first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Live Data (CSV)",
            initialfile="morphogenesis_series.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            import csv
            # Write time/acceptance/residual series as CSV
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "time_us", "accept_ratio", "residual_norm"])
                for i, t in enumerate(self._live_times):
                    ar = self._live_accept[i] if i &lt; len(self._live_accept) else 0.0
                    rn = float(self._live_resid[i]) if i &lt; len(self._live_resid) else 0.0
                    w.writerow([i, f"{t:.3f}", f"{ar:.6f}", f"{rn:.6e}"])
            # Snapshot JSON of current parameters alongside CSV
            snap = {
                "grid": int(self.var_grid.get()),
                "iterations": int(self.var_iters.get()),
                "geom_mode": self.var_geom.get(),
                "trust_radius": float(self.var_trust.get()),
                "hysteresis": int(self.var_hyst.get()),
                "period_us": int(self.var_period_us.get()),
                "fp_cycles": int(self.var_fp_cycles.get()),
            }
            with open(os.path.splitext(path)[0] + ".json", "w") as jf:
                json.dump(snap, jf, indent=2, sort_keys=True)
            messagebox.showinfo("Export", f"Saved data to {path} and snapshot JSON")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _on_fp_probe(self) -> None:
        # Display fast-plane KPIs for the selected grid and cycles
        try:
            N = int(self.var_grid.get())
            cycles = int(self.var_fp_cycles.get())
        except Exception:
            messagebox.showerror("Invalid input", "Grid/FP cycles must be integers.")
            return
        try:
            kpi = _fast_plane_probe(N=N, cycles=cycles)
            msg = (
                f"N={kpi.get('N')} cycles={kpi.get('cycles')}\n"
                f"produced_flits_total={kpi.get('produced_flits_total')}\n"
                f"served_mem_requests={kpi.get('served_mem_requests')}\n"
                f"avg_mc_latency={float(kpi.get('avg_mc_latency', 0.0)):.2f}  "
                f"p95_mc_latency={float(kpi.get('p95_mc_latency', 0.0)):.2f}\n"
                f"power_total_energy={float(kpi.get('power_total_energy', 0.0)):.3f}  "
                f"max_temp_any_tile={float(kpi.get('max_temp_any_tile', 0.0)):.2f}"
            )
            messagebox.showinfo("Fast-plane KPIs", msg)
        except Exception as e:
            messagebox.showerror("FP Probe failed", str(e))

    def _apply_preset(self, name: str) -> None:
        if name == "quick":
            self.var_iters.set(20)
            self.var_fp_cycles.set(500)
            self.var_status.set("Preset applied: quick")
        elif name == "extended":
            self.var_iters.set(200)
            self.var_fp_cycles.set(5000)
            self.var_status.set("Preset applied: extended")
        else:
            self.var_status.set(f"Preset not recognized: {name}")
    def _on_sweep_report(self) -> None:
        if self._running:
            messagebox.showwarning("Busy", "Stop live morphogenesis before running a sweep.")
            return
        try:
            N = int(self.var_grid.get())
            iters = int(self.var_iters.get())
            fp_cycles = int(self.var_fp_cycles.get())
            period = int(self.var_period_us.get())
        except Exception:
            messagebox.showerror("Invalid input", "Parameters must be integers.")
            return

        # Choose output path
        path = filedialog.asksaveasfilename(
            title="Save Morphogenesis Report",
            initialfile="e4_morphogenesis_report.md",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("All Files", "*.*")],
        )
        if not path:
            return

        self.var_status.set("Running sweep...")
        self.update_idletasks()
        try:
            grids = [4, 8, 16, 32]  # full set (user-selected N influences live view; report uses canonical set)
            sweep = run_morphogenesis_sweep(grids=grids, iterations=iters, fp_cycles=fp_cycles, slow_loop_period_us=period)
            out_path = write_morphogenesis_report(sweep, path)
            self.var_status.set(f"Wrote report: {out_path}")
            messagebox.showinfo("Report", f"Wrote report: {out_path}")
        except Exception as e:
            self.var_status.set("Sweep failed")
            messagebox.showerror("Sweep failed", str(e))

    # ---------------- State persistence ----------------
    def _load_state(self) -> None:
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r") as f:
                    s = json.load(f)
                self.var_grid.set(int(s.get("grid", 8)))
                self.var_iters.set(int(s.get("iters", 50)))
                self.var_fp_cycles.set(int(s.get("fp_cycles", 2000)))
                self.var_period_us.set(int(s.get("period_us", 1_000_000)))
                self.var_geom.set(str(s.get("geom_mode", "baseline")))
                self.var_trust.set(float(s.get("trust_radius", 0.25)))
                self.var_hyst.set(int(s.get("hysteresis", 2)))
        except Exception:
            pass

    def _save_state(self) -> None:
        try:
            s = {
                "grid": int(self.var_grid.get()),
                "iters": int(self.var_iters.get()),
                "fp_cycles": int(self.var_fp_cycles.get()),
                "period_us": int(self.var_period_us.get()),
                "geom_mode": self.var_geom.get(),
                "trust_radius": float(self.var_trust.get()),
                "hysteresis": int(self.var_hyst.get()),
            }
            with open(self._state_path, "w") as f:
                json.dump(s, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_state()
        self.destroy()

    # ---------------- Worker logic ----------------

    def _live_worker(self, N: int, iterations: int) -> None:
        try:
            # Build initial state
            H = W = N
            g = _mk_identity_metric(H, W)
            arts = syn_artifacts(H, W)
            U = arts["U"].tolist() if hasattr(arts["U"], "tolist") else arts["U"]
            J = arts["J"].tolist() if hasattr(arts["J"], "tolist") else arts["J"]
            B = arts["B"]

            geom_label = self.var_geom.get()
            geom_cfg = _geom_cfg_from_label(geom_label)
            if self.var_geom.get() == "custom":
                geom_cfg = GeometryConfig(trust_radius=float(self.var_trust.get()), hysteresis=int(self.var_hyst.get()))

            # Live loop
            field_cfg = FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-5, boundary="neumann", grad_clip=0.0)
            acc_count = 0
            for step in range(iterations):
                if not self._running:
                    break

                # Field solve
                t0 = time.perf_counter()
                field_out = solve_field(g, J, field_cfg)
                phi = field_out["phi"]
                grad = field_out["grad"]

                # Geometry update
                gcfg = geom_cfg or GeometryConfig()
                geom_out = update_geometry(g, phi, grad, U, J, B, gcfg)
                g_next = geom_out["g_next"]
                tri = geom_out["meta"]

                # Control apply timing (end-to-end)
                tel = {"max_vc_depth": 0, "temp_max": 50.0, "power_proxy_avg": 0.5}
                _ = control_apply_cycle(g, {"U": U, "J": J, "B": B}, telemetry=tel, geometry_cfg=geom_cfg, rollback_on_fail=True)
                t1 = time.perf_counter()
                dt_us = (t1 - t0) * 1_000_000.0

                # Update buffers
                self._live_times.append(dt_us)
                if bool(tri.get("accepted", False)):
                    acc_count += 1
                acc_ratio = float(acc_count / max(1, (step + 1)))
                self._live_accept.append(acc_ratio)
                self._live_resid.append(float(tri.get("residual_norm", 0.0)))

                # Update heatmap
                self._update_phi(phi)
                # Update charts
                self._update_charts(self._live_times, self._live_accept)
                # Update summary table
                self._update_summary_table(self._live_times, acc_ratio)

                # Advance metric
                g = g_next

                # Small sleep to keep UI responsive
                self.after(1)

            self.var_status.set("Live run complete")
        except Exception as e:
            self.var_status.set("Live run failed")
            messagebox.showerror("Live morphogenesis failed", str(e))
        finally:
            self._running = False

    # ---------------- Plot helpers ----------------

    def _update_phi(self, phi: Any) -> None:
        # Render phi as heatmap
        self.ax_phi.clear()
        self.im_phi = self.ax_phi.imshow(phi, cmap="viridis", aspect="auto")
        self.ax_phi.set_title("Φ (field)")
        self.canvas_left.draw_idle()

    def _update_charts(self, times: List[float], accepts: List[float]) -> None:
        self.ax_time.clear()
        self.ax_acc.clear()
        self.ax_time.plot(times, color="tab:blue")
        self.ax_time.set_title("Control Apply Time (us)")
        self.ax_time.set_ylabel("us")
        self.ax_acc.plot(accepts, color="tab:green")
        self.ax_acc.set_title("Trust-region Acceptance Ratio")
        self.ax_acc.set_ylim(0, 1.0)
        # Plot residual on secondary axis
        self.ax_acc2.clear()
        self.ax_acc2.plot(accepts if len(self._live_resid) != len(accepts) else self._live_resid, color="tab:orange")
        self.ax_acc2.set_ylabel("residual_norm", color="tab:orange")
        self.fig_right.tight_layout()
        self.canvas_right.draw_idle()

    # ---------------- Comparison helpers ----------------
    def _on_load_summary(self, slot: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Load Summary {slot}",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            import json as _json
            with open(path, "r") as f:
                data = _json.load(f)
            if slot == 'A':
                self._summary_a = data
            else:
                self._summary_b = data
            messagebox.showinfo("Loaded", f"Loaded summary {slot} from {path}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    @staticmethod
    def _extract_baseline_p95(summary: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        grids: List[str] = []
        p95s: List[float] = []
        results = summary.get("results", {})
        if not isinstance(results, dict):
            return grids, p95s
        for grid_key in sorted(results.keys(), key=lambda s: (len(s), s)):
            rec = results.get(grid_key, {})
            ctl = rec.get("control", {})
            base = ctl.get("baseline", {})
            stats = base.get("stats", {})
            p95 = float(stats.get("p95_us", 0.0))
            grids.append(grid_key)
            p95s.append(p95)
        return grids, p95s

    def _on_compare_summaries(self) -> None:
        if self._summary_a is None or self._summary_b is None:
            messagebox.showwarning("Missing summaries", "Load Summary A and B first.")
            return
        grids_a, p95_a = self._extract_baseline_p95(self._summary_a)
        grids_b, p95_b = self._extract_baseline_p95(self._summary_b)
        if grids_a != grids_b:
            messagebox.showwarning("Grid mismatch", "Summaries use different grid sets; cannot compare.")
            return
        # Create a simple comparison window with a bar chart
        win = tk.Toplevel(self)
        win.title("Baseline p95 Comparison (A vs B)")
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        try:
            import numpy as np  # local import
            x = np.arange(len(grids_a))
            w = 0.35
            ax.bar(x - w/2, p95_a, width=w, label="A")
            ax.bar(x + w/2, p95_b, width=w, label="B")
            ax.set_xticks(x)
            ax.set_xticklabels(grids_a)
        except Exception:
            # Fallback without numpy: place bars at integer positions
            x = list(range(len(grids_a)))
            ax.bar([i - 0.2 for i in x], p95_a, width=0.4, label="A")
            ax.bar([i + 0.2 for i in x], p95_b, width=0.4, label="B")
            ax.set_xticks(x)
            ax.set_xticklabels(grids_a)
        ax.set_ylabel("p95_us")
        ax.set_title("Baseline p95 by grid")
        ax.legend()
        canv = FigureCanvasTkAgg(fig, master=win) if FigureCanvasTkAgg else None
        if canv:
            canv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            fig.tight_layout()
            canv.draw_idle()

class AppMain:
    @staticmethod
    def run() -> None:
        # Guard if Tk backend is unavailable
        if FigureCanvasTkAgg is None or not _tk_available():
            raise RuntimeError(
                "Tkinter backend not available; cannot launch GUI. "
                "Run headless benchmarks instead:\n"
                "  .venv/bin/python -m harness.e4_bench --grid 8x8 --iterations 50 --md harness/e4_bench_report.md\n"
                "  .venv/bin/python -m harness.e4_morphogenesis_bench --grids 4,8,16,32 --iterations 100 --fp-cycles 2000 --period-us 1000000 --out harness/reports/e4_morphogenesis_report.md"
            )
        app = MorphogenesisGUI()
        app.mainloop()


if __name__ == "__main__":
    AppMain.run()