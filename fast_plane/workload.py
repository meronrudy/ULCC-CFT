from __future__ import annotations

"""
E1d: Minimal deterministic workload loader.

Inputs (scenario: dict):
- topology:
  - Either {"N": int} for NxN mesh validation, or {"mesh": (w, h)}.
- scheduler: optional {"quantum": int, "default_tile": int}
- tasks: List of {
    "task_id": int,
    "priority": int,
    "affinity": int | (x,y) | [int|(x,y), ...],
    "dst": int | (x,y) | [int|(x,y), ...],
    "vc": "MAIN"|"ESC" (default "MAIN"),
    "message_size_flits": int,
    "rate_tokens_per_cycle": float,
    "burst_size_flits": int,
    "enable": bool (default True),
    "demand_hint": float (optional)
  }
- consumers: optional List of {
    "tile": int | (x,y),
    "service_rate_flits_per_cycle": float (default 16.0),
    "sink_latency_cycles": int (default 0)
  }
- cache_miss: optional List of {
    "tile": int | (x,y),
    "mpki": float,
    "ipc": float,
    "message_size_flits": int (default 4),
    "vc": "MAIN"|"ESC" (default "MAIN"),
    "mc_tile": int | (x,y) (default (0,0)),
    "enable": bool (default True)
  }
- mc: optional {
    "tile": int | (x,y),
    "bank_count": int (default 8),
    "channel_count": int (default 2),
    "rows_per_bank": int (default 64),
    "window_size": int (default 8),
    "t_row_hit": int (default 40),
    "t_row_miss": int (default 200),
    "t_bus": int (default 20),
    "mode": "FRFCFS"|"FIFO" (default "FRFCFS")
  }
- scenario_id: optional str

Outputs:
- Returns the attached Scheduler instance. Side-effects: registers producers, consumers, cache miss processes,
  memory controller, and the scheduler on the provided NoC.

Determinism guarantees:
- No randomness; stable registration order; stable submit_index order.
- Producers are parameterized statically and only gated via scheduler (_sched_enabled).
"""

from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

from fast_plane.types import VC, Coord
from fast_plane.cores import TokenBucketProducer, Consumer
from fast_plane.cache import CacheMissProcess
from fast_plane.mc import MemoryController, MCConfig
from fast_plane.scheduler import Scheduler, Task


TileLike = Union[int, Tuple[int, int]]
DstLike = Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]]


def _to_tile_id(noc: "NoC", v: TileLike) -> int:
    if isinstance(v, int):
        return v
    x, y = int(v[0]), int(v[1])
    return noc.mesh.tile_id(Coord(x, y))


def _dst_to_selection(noc: "NoC", dst: DstLike):
    if isinstance(dst, (list, tuple)) and not (isinstance(dst, tuple) and len(dst) == 2 and all(isinstance(x, int) for x in dst)):
        # sequence of entries
        out: List[int] = []
        for it in dst:  # type: ignore[assignment]
            if isinstance(it, int):
                out.append(int(it))
            else:
                out.append(_to_tile_id(noc, it))
        if len(out) == 0:
            raise ValueError("dst list must not be empty")
        return out
    # single entry
    if isinstance(dst, int):
        return int(dst)
    return _to_tile_id(noc, dst)


def _vc_from_str(s: Optional[str]) -> VC:
    if s is None:
        return VC.MAIN
    s = s.upper()
    if s == "MAIN":
        return VC.MAIN
    if s == "ESC":
        return VC.ESC
    raise ValueError("vc must be 'MAIN' or 'ESC'")


def _validate_topology(noc: "NoC", topo: Dict[str, Any]) -> None:
    if not topo:
        return
    if "N" in topo:
        N = int(topo["N"])
        if noc.mesh.width != N or noc.mesh.height != N:
            raise AssertionError(f"Scenario topology N={N} does not match NoC mesh {noc.mesh.width}x{noc.mesh.height}")
    elif "mesh" in topo:
        w, h = int(topo["mesh"][0]), int(topo["mesh"][1])
        if noc.mesh.width != w or noc.mesh.height != h:
            raise AssertionError(f"Scenario topology mesh=({w},{h}) does not match NoC mesh {noc.mesh.width}x{noc.mesh.height}")
    # else: ignore (validation optional)


def load_workload(noc: "NoC", scenario: Dict[str, Any]) -> Scheduler:
    # Topology validation (optional)
    _validate_topology(noc, scenario.get("topology", {}) or {})

    # Optional scenario id
    scenario_id = scenario.get("scenario_id") or scenario.get("id")
    if scenario_id is not None and hasattr(noc, "set_scenario_id"):
        noc.set_scenario_id(str(scenario_id))  # type: ignore[attr-defined]

    # Consumers (non-MC)
    for cons in scenario.get("consumers", []) or []:
        tile = _to_tile_id(noc, cons["tile"])
        sr = float(cons.get("service_rate_flits_per_cycle", 16.0))
        lat = int(cons.get("sink_latency_cycles", 0))
        c = Consumer(service_rate_flits_per_cycle=sr, sink_latency_cycles=lat)
        noc.register_consumer(tile, c)

    # Memory controller (single in minimal loader; extendable)
    if "mc" in scenario and scenario["mc"]:
        mc_cfg = scenario["mc"]
        tile = _to_tile_id(noc, mc_cfg["tile"])
        cfg = MCConfig(
            bank_count=int(mc_cfg.get("bank_count", 8)),
            channel_count=int(mc_cfg.get("channel_count", 2)),
            rows_per_bank=int(mc_cfg.get("rows_per_bank", 64)),
            window_size=int(mc_cfg.get("window_size", 8)),
            t_row_hit=int(mc_cfg.get("t_row_hit", 40)),
            t_row_miss=int(mc_cfg.get("t_row_miss", 200)),
            t_bus=int(mc_cfg.get("t_bus", 20)),
            mode=str(mc_cfg.get("mode", "FRFCFS")),
        )
        mc = MemoryController(cfg)
        noc.register_memory_controller(tile, mc)

    # Cache-miss processes
    for cm in scenario.get("cache_miss", []) or []:
        tile = _to_tile_id(noc, cm["tile"])
        mpki = float(cm["mpki"])
        ipc = float(cm["ipc"])
        msg_size = int(cm.get("message_size_flits", 4))
        vc = _vc_from_str(cm.get("vc"))
        mc_tile = cm.get("mc_tile", (0, 0))
        mc_xy = mc_tile if isinstance(mc_tile, tuple) else noc.mesh.coord_of(int(mc_tile))
        enable = bool(cm.get("enable", True))
        proc = CacheMissProcess(
            mpki=mpki,
            ipc=ipc,
            message_size_flits=msg_size,
            vc=vc,
            mc_tile=(int(mc_xy[0]), int(mc_xy[1])) if isinstance(mc_xy, tuple) else (mc_xy.x, mc_xy.y),
            enable=enable,
        )
        noc.register_cache_process(tile, proc)

    # Scheduler (quantum config)
    sched_cfg = scenario.get("scheduler", {}) or {}
    quantum = int(sched_cfg.get("quantum", 8))
    default_tile = int(sched_cfg.get("default_tile", 0))
    sched = Scheduler(quantum_cycles=quantum, default_tile_id=default_tile)

    # Tasks and producers
    for t in scenario.get("tasks", []) or []:
        task_id = int(t["task_id"])
        pri = int(t.get("priority", 0))
        affinity = t.get("affinity")
        # Normalize affinity into list of tile-like entries
        if affinity is None:
            aff_list: List[TileLike] = []
        elif isinstance(affinity, (list, tuple)) and not (isinstance(affinity, tuple) and len(affinity) == 2 and all(isinstance(x, int) for x in affinity)):
            aff_list = list(affinity)  # type: ignore[list-item]
        else:
            aff_list = [affinity]  # type: ignore[list-item]

        demand = t.get("demand_hint", None)
        enable_flag = bool(t.get("enable", True))
        task = Task(
            task_id=task_id,
            priority=pri,
            affinity=aff_list if aff_list else None,
            demand_hint=float(demand) if demand is not None else None,
            enabled=enable_flag,
        )
        sched.register_task(task)

        # Producer parameters
        msg_sz = int(t["message_size_flits"])
        rate = float(t["rate_tokens_per_cycle"])
        burst = int(t["burst_size_flits"])
        vc = _vc_from_str(t.get("vc"))
        dst_sel = _dst_to_selection(noc, t["dst"])

        # Create one producer per affinity tile (or default tile if none)
        if aff_list:
            tiles = [_to_tile_id(noc, a) for a in aff_list]
        else:
            tiles = [default_tile]

        for tile_id in tiles:
            prod = TokenBucketProducer(
                rate_tokens_per_cycle=rate,
                burst_size_flits=burst,
                message_size_flits=msg_sz,
                dst_selection=dst_sel,
                vc=vc,
                rng_seed=0,
            )
            # Register to NoC and bind to scheduler; start disabled for gating
            # NoC registration attaches producer to tile deterministically (insertion order preserved).
            noc.register_producer(tile_id, prod)
            setattr(prod, "_sched_enabled", False)
            sched.bind_task_producer(task_id, tile_id, prod)

    # Finally attach scheduler to NoC (this calls bootstrap() to program initial window)
    noc.register_scheduler(sched)

    return sched
# ---- E1g: JSON config loader ----
import json
import os


def load_workload_from_json(path: str) -> Dict[str, Any]:
    """
    Load a scenario JSON file and return a normalized scenario dict compatible with load_workload().

    Validation rules (top-level):
      - Allowed keys: {"topology", "tasks", "consumers", "cache_miss", "mc", "scheduler", "scenario_id", "rng_seed", "power", "thermal"}
      - Required: "topology" and at least one of {"tasks", "cache_miss"}; others optional.
      - topology: either {"N": int} or {"mesh": [w, h] | (w, h)}
      - scheduler: optional; defaults applied: {"quantum": 8, "default_tile": 0}
      - mc, consumers, cache_miss: optional; defaults applied per fields in E1 loader when absent
      - tasks: list of producers; for each task, required:
          {"task_id", "dst", "message_size_flits", "rate_tokens_per_cycle", "burst_size_flits"}
        Optional: {"priority"=0, "vc"="MAIN", "affinity"=None, "enable"=True, "demand_hint"=None}

    Deterministic defaults align with sim/API_SURFACE.md for Phase A:
      - scheduler.quantum = 8
      - scheduler.default_tile = 0
      - consumer.service_rate_flits_per_cycle = 16.0
      - consumer.sink_latency_cycles = 0
      - cache_miss.message_size_flits = 4
      - cache_miss.vc = "MAIN"
      - cache_miss.mc_tile = (0, 0)
      - cache_miss.enable = True
      - mc window_size=8, t_row_hit=40, t_row_miss=200, t_bus=20, bank_count=8, channel_count=2, rows_per_bank=64, mode="FRFCFS"

    Raises:
      ValueError with clear message for malformed sections or missing required fields.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("load_workload_from_json: path must be a non-empty string")
    if not os.path.exists(path):
        raise ValueError(f"load_workload_from_json: file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise ValueError(f"load_workload_from_json: failed to parse JSON: {e}")

    if not isinstance(raw, dict):
        raise ValueError("scenario root must be a JSON object")

    allowed_keys = {
        "topology", "tasks", "consumers", "cache_miss", "mc", "scheduler",
        "scenario_id", "rng_seed", "power", "thermal"
    }
    extra = sorted([k for k in raw.keys() if k not in allowed_keys])
    if extra:
        # Be permissive but warn via exception per spec for E1g
        raise ValueError(f"unknown top-level keys in scenario: {extra}")

    if "topology" not in raw or not raw["topology"]:
        raise ValueError("scenario.topology is required")

    topology = raw["topology"]
    if not isinstance(topology, dict):
        raise ValueError("scenario.topology must be an object")
    norm_topo: Dict[str, Any] = {}
    if "N" in topology:
        try:
            N = int(topology["N"])
        except Exception:
            raise ValueError("topology.N must be an integer")
        if N <= 0:
            raise ValueError("topology.N must be > 0")
        norm_topo["N"] = N
    elif "mesh" in topology:
        mesh = topology["mesh"]
        if (not isinstance(mesh, (list, tuple))) or len(mesh) != 2:
            raise ValueError("topology.mesh must be [w, h] or (w, h)")
        try:
            w = int(mesh[0]); h = int(mesh[1])
        except Exception:
            raise ValueError("topology.mesh entries must be integers")
        if w <= 0 or h <= 0:
            raise ValueError("topology.mesh width/height must be > 0")
        norm_topo["mesh"] = [w, h]
    else:
        raise ValueError("topology must contain either 'N' or 'mesh'")

    # Scheduler defaults
    sched = raw.get("scheduler") or {}
    if not isinstance(sched, dict):
        raise ValueError("scheduler must be an object when present")
    quantum = int(sched.get("quantum", 8))
    default_tile = int(sched.get("default_tile", 0))
    if quantum < 1:
        raise ValueError("scheduler.quantum must be >= 1")
    norm_sched = {"quantum": quantum, "default_tile": default_tile}

    # Consumers
    consumers_in = raw.get("consumers") or []
    if not isinstance(consumers_in, list):
        raise ValueError("consumers must be a list when present")
    norm_consumers: List[Dict[str, Any]] = []
    for i, c in enumerate(consumers_in):
        if not isinstance(c, dict):
            raise ValueError(f"consumers[{i}] must be an object")
        if "tile" not in c:
            raise ValueError(f"consumers[{i}].tile is required")
        tile = c["tile"]
        if isinstance(tile, (list, tuple)) and len(tile) == 2:
            tile = (int(tile[0]), int(tile[1]))
        elif isinstance(tile, int):
            tile = int(tile)
        else:
            raise ValueError(f"consumers[{i}].tile must be int or (x,y)")
        sr = float(c.get("service_rate_flits_per_cycle", 16.0))
        lat = int(c.get("sink_latency_cycles", 0))
        norm_consumers.append({"tile": tile, "service_rate_flits_per_cycle": sr, "sink_latency_cycles": lat})

    # MC
    mc_norm: Optional[Dict[str, Any]] = None
    if raw.get("mc"):
        mc = raw["mc"]
        if not isinstance(mc, dict):
            raise ValueError("mc must be an object")
        if "tile" not in mc:
            raise ValueError("mc.tile is required")
        tile = mc["tile"]
        if isinstance(tile, (list, tuple)) and len(tile) == 2:
            tile = (int(tile[0]), int(tile[1]))
        elif isinstance(tile, int):
            tile = int(tile)
        else:
            raise ValueError("mc.tile must be int or (x,y)")
        mc_norm = {
            "tile": tile,
            "bank_count": int(mc.get("bank_count", 8)),
            "channel_count": int(mc.get("channel_count", 2)),
            "rows_per_bank": int(mc.get("rows_per_bank", 64)),
            "window_size": int(mc.get("window_size", 8)),
            "t_row_hit": int(mc.get("t_row_hit", 40)),
            "t_row_miss": int(mc.get("t_row_miss", 200)),
            "t_bus": int(mc.get("t_bus", 20)),
            "mode": str(mc.get("mode", "FRFCFS")),
        }

    # Cache miss processes
    cache_in = raw.get("cache_miss") or []
    if not isinstance(cache_in, list):
        raise ValueError("cache_miss must be a list when present")
    norm_cache: List[Dict[str, Any]] = []
    for i, cm in enumerate(cache_in):
        if not isinstance(cm, dict):
            raise ValueError(f"cache_miss[{i}] must be an object")
        for req in ("tile", "mpki", "ipc"):
            if req not in cm:
                raise ValueError(f"cache_miss[{i}].{req} is required")
        tile = cm["tile"]
        if isinstance(tile, (list, tuple)) and len(tile) == 2:
            tile = (int(tile[0]), int(tile[1]))
        elif isinstance(tile, int):
            tile = int(tile)
        else:
            raise ValueError(f"cache_miss[{i}].tile must be int or (x,y)")
        mc_tile = cm.get("mc_tile", (0, 0))
        if isinstance(mc_tile, (list, tuple)) and len(mc_tile) == 2:
            mc_tile = (int(mc_tile[0]), int(mc_tile[1]))
        elif isinstance(mc_tile, int):
            mc_tile = int(mc_tile)
        else:
            raise ValueError(f"cache_miss[{i}].mc_tile must be int or (x,y)")
        norm_cache.append({
            "tile": tile,
            "mpki": float(cm["mpki"]),
            "ipc": float(cm["ipc"]),
            "message_size_flits": int(cm.get("message_size_flits", 4)),
            "vc": str(cm.get("vc", "MAIN")),
            "mc_tile": mc_tile,
            "enable": bool(cm.get("enable", True)),
        })

    # Tasks (producers)
    tasks_in = raw.get("tasks") or []
    if not isinstance(tasks_in, list):
        raise ValueError("tasks must be a list when present")
    norm_tasks: List[Dict[str, Any]] = []
    for i, t in enumerate(tasks_in):
        if not isinstance(t, dict):
            raise ValueError(f"tasks[{i}] must be an object")
        required = ["task_id", "dst", "message_size_flits", "rate_tokens_per_cycle", "burst_size_flits"]
        for req in required:
            if req not in t:
                raise ValueError(f"tasks[{i}].{req} is required")
        # affinity can be int, (x,y), or list of such
        aff = t.get("affinity", None)
        if aff is None:
            norm_aff = None
        elif isinstance(aff, (list, tuple)) and not (isinstance(aff, tuple) and len(aff) == 2 and all(isinstance(x, int) for x in aff)):
            # list-like; normalize inner pairs to tuples
            lst = []
            for ent in aff:  # type: ignore[assignment]
                if isinstance(ent, int):
                    lst.append(int(ent))
                elif isinstance(ent, (list, tuple)) and len(ent) == 2:
                    lst.append((int(ent[0]), int(ent[1])))
                else:
                    raise ValueError(f"tasks[{i}].affinity entries must be int or (x,y)")
            norm_aff = lst
        else:
            # single entry
            if isinstance(aff, int):
                norm_aff = int(aff)
            elif isinstance(aff, (list, tuple)) and len(aff) == 2:
                norm_aff = (int(aff[0]), int(aff[1]))
            else:
                raise ValueError(f"tasks[{i}].affinity must be int, (x,y), or list")
        # dst normalize: int, (x,y), or list of such
        dst = t["dst"]
        if isinstance(dst, (list, tuple)) and not (isinstance(dst, tuple) and len(dst) == 2 and all(isinstance(x, int) for x in dst)):
            lst = []
            for ent in dst:  # type: ignore[assignment]
                if isinstance(ent, int):
                    lst.append(int(ent))
                elif isinstance(ent, (list, tuple)) and len(ent) == 2:
                    lst.append((int(ent[0]), int(ent[1])))
                else:
                    raise ValueError(f"tasks[{i}].dst entries must be int or (x,y)")
            norm_dst = lst
        else:
            if isinstance(dst, int):
                norm_dst = int(dst)
            elif isinstance(dst, (list, tuple)) and len(dst) == 2:
                norm_dst = (int(dst[0]), int(dst[1]))
            else:
                raise ValueError(f"tasks[{i}].dst must be int, (x,y), or list")
        norm_tasks.append({
            "task_id": int(t["task_id"]),
            "priority": int(t.get("priority", 0)),
            "affinity": norm_aff,
            "dst": norm_dst,
            "vc": str(t.get("vc", "MAIN")),
            "message_size_flits": int(t["message_size_flits"]),
            "rate_tokens_per_cycle": float(t["rate_tokens_per_cycle"]),
            "burst_size_flits": int(t["burst_size_flits"]),
            "enable": bool(t.get("enable", True)),
            "demand_hint": (float(t["demand_hint"]) if "demand_hint" in t and t["demand_hint"] is not None else None),
        })

    if not norm_tasks and not norm_cache and not norm_consumers and mc_norm is None:
        # Require at least some traffic source; consumers alone are allowed but not useful.
        raise ValueError("scenario must define at least one of: tasks, cache_miss, mc or consumers")

    scenario_id = raw.get("scenario_id")
    if scenario_id is not None and not isinstance(scenario_id, str):
        raise ValueError("scenario_id must be a string when present")
    rng_seed = int(raw.get("rng_seed", 0))

    # Pass-through optional model hints (power/thermal) if present; runner may use them.
    out: Dict[str, Any] = {
        "topology": norm_topo,
        "scheduler": norm_sched,
        "tasks": norm_tasks,
        "consumers": norm_consumers,
        "cache_miss": norm_cache,
        "mc": mc_norm,
        "scenario_id": scenario_id,
        "rng_seed": rng_seed,
    }
    if raw.get("power") is not None:
        out["power"] = raw["power"]
    if raw.get("thermal") is not None:
        out["thermal"] = raw["thermal"]
    return out