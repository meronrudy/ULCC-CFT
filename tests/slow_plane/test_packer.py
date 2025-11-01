# Ensure project root is on sys.path for direct pytest invocation of a nested module
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import numpy as np

from slow_plane.packer import PackerConfig, make_reconfig_pack
from slow_plane.packer.make import _canonical_json_bytes, _crc32c  # test internals for determinism
from slow_plane.field.solver import solve_field, FieldConfig
from slow_plane.geometry.update import update_geometry, GeometryConfig


def _make_identity_metric(H: int, W: int) -> np.ndarray:
    g = np.zeros((H, W, 2, 2), dtype=np.float64)
    g[..., 0, 0] = 1.0
    g[..., 1, 1] = 1.0
    return g


def _B_from_grad_dirs(grad: np.ndarray) -> dict:
    # Construct B hints from -∇Phi projections (non-negative)
    gx = grad[..., 0]
    gy = grad[..., 1]
    B_N = np.maximum(gy, 0.0)
    B_S = np.maximum(-gy, 0.0)
    B_E = np.maximum(-gx, 0.0)
    B_W = np.maximum(gx, 0.0)
    return {"N": B_N, "S": B_S, "E": B_E, "W": B_W}


def _build_inputs(H: int = 6, W: int = 6):
    # Metric
    g = _make_identity_metric(H, W)
    # Source J: centered impulse
    J = np.zeros((H, W), dtype=np.float64)
    J[H // 2, W // 2] = 1.0
    # Field solve (steady-state for determinism)
    fcfg = FieldConfig(method="cg", max_cg_iters=200, cg_tol=1e-8, boundary="neumann")
    field = solve_field(g, J, fcfg)
    phi = field["phi"]
    grad = field["grad"]
    # Attribution proxy U from |phi| normalized
    if np.max(np.abs(phi)) > 0:
        U = np.abs(phi) / np.max(np.abs(phi))
    else:
        U = np.zeros_like(phi)
    # Flux B from grad dirs
    B = _B_from_grad_dirs(grad)
    # Geometry meta via update
    gcfg = GeometryConfig()
    geom = update_geometry(g, phi, grad, U, J, B, gcfg)
    geom_meta = geom["meta"]
    return g, phi, grad, U, J, B, geom_meta


def _crc_over_pack_without_crc(pack: dict) -> int:
    tmp = dict(pack)
    tmp.pop("crc32c", None)
    payload = _canonical_json_bytes(tmp)
    return _crc32c(payload)


def test_schema_and_shapes():
    H, W = 6, 6
    g, phi, grad, U, J, B, geom_meta = _build_inputs(H, W)
    cfg = PackerConfig(dvfs_levels=[0, 1, 2], n_cpus=16)

    pack = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg)

    # Keys present
    for k in (
        "version",
        "noc_tables",
        "link_weights",
        "mc_policy_words",
        "cat_masks",
        "cpu_affinities",
        "numa_policies",
        "dvfs_states",
        "trust_region_meta",
        "crc32c",
    ):
        assert k in pack, f"missing key {k}"

    assert pack["version"] == 1

    weights = pack["noc_tables"]["weights"]
    link_w = pack["link_weights"]
    mc = pack["mc_policy_words"]
    cat = pack["cat_masks"]
    cpu = pack["cpu_affinities"]
    numa = pack["numa_policies"]
    dvfs = pack["dvfs_states"]
    tr = pack["trust_region_meta"]

    # Shapes
    assert isinstance(weights, np.ndarray) and weights.shape == (H, W, 4)
    assert isinstance(link_w, np.ndarray) and link_w.shape == (H, W, 4)
    assert isinstance(mc, np.ndarray) and mc.shape == (H, W)
    assert isinstance(cat, np.ndarray) and cat.shape == (H, W)
    assert isinstance(cpu, np.ndarray) and cpu.shape == (H, W)
    assert isinstance(numa, np.ndarray) and numa.shape == (H, W)
    assert isinstance(dvfs, np.ndarray) and dvfs.shape == (H, W)

    # Dtypes
    assert weights.dtype == np.float32
    assert link_w.dtype == np.float32
    assert mc.dtype == np.float32
    assert cat.dtype == np.int32
    assert cpu.dtype == np.int32
    assert dvfs.dtype == np.int32
    # numa is dtype object (strings)
    assert numa.dtype == object

    # Weights bounds and normalization
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0 + 1e-6)
    row_sums = np.sum(weights.astype(np.float64), axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    # MC priorities in bounds
    assert np.all(mc >= cfg.mc_min_priority - 1e-9)
    assert np.all(mc <= cfg.mc_max_priority + 1e-9)

    # CAT masks are ints within bit width
    max_mask = (1 << cfg.cat_num_ways) - 1
    assert np.issubdtype(cat.dtype, np.integer)
    assert np.all(cat >= 0)
    assert np.all(cat <= max_mask)

    # CPU ids within range
    assert np.all(cpu >= 0)
    assert np.all(cpu < cfg.n_cpus)

    # NUMA policies fixed string
    assert np.all(numa == "local_first")

    # DVFS values in allowed set
    allowed = set(cfg.dvfs_levels)
    assert all(int(v) in allowed for v in dvfs.ravel())

    # Trust region meta mirrored
    for k in ("accepted", "accept_ratio", "residual_norm", "trust_radius", "hysteresis_left"):
        assert k in tr
        assert tr[k] == geom_meta[k]

    # CRC present and matches canonical payload excluding crc itself
    crc = int(pack["crc32c"])
    recomputed = _crc_over_pack_without_crc(pack)
    assert crc == recomputed


def test_bounds_and_clipping():
    H, W = 4, 5
    g = _make_identity_metric(H, W)
    phi = np.zeros((H, W), dtype=np.float64)
    # Extreme gradient favoring East and South
    grad = np.zeros((H, W, 2), dtype=np.float64)
    grad[..., 0] = -1e9  # large negative gx -> E preference
    grad[..., 1] = -1e9  # large negative gy -> S preference
    # Extreme B also to E and S
    B = {
        "N": np.zeros((H, W), dtype=np.float64),
        "S": np.full((H, W), 1e9, dtype=np.float64),
        "E": np.full((H, W), 1e9, dtype=np.float64),
        "W": np.zeros((H, W), dtype=np.float64),
    }
    U = np.zeros((H, W), dtype=np.float64)
    J = np.zeros((H, W), dtype=np.float64)

    # Geometry meta (from a benign update)
    gcfg = GeometryConfig()
    geom = update_geometry(g, phi, grad*0.0, U, J, B, gcfg)
    geom_meta = geom["meta"]

    clip_min = 0.05
    cfg = PackerConfig(dvfs_levels=[0, 1], routing_clip_min=clip_min)

    pack = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg)
    weights = pack["noc_tables"]["weights"].astype(np.float64)

    assert np.all(weights >= clip_min - 1e-6)
    row_sums = np.sum(weights, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_determinism():
    H, W = 6, 6
    g, phi, grad, U, J, B, geom_meta = _build_inputs(H, W)
    cfg = PackerConfig(dvfs_levels=[0, 1, 2, 3], n_cpus=8)

    pack1 = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg)
    pack2 = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg)

    # Byte-identical canonical JSON with CRC included
    b1 = _canonical_json_bytes(pack1)
    b2 = _canonical_json_bytes(pack2)
    assert b1 == b2

    # CRC matches recomputation
    crc1 = int(pack1["crc32c"])
    crc2 = int(pack2["crc32c"])
    assert crc1 == crc2
    assert crc1 == _crc_over_pack_without_crc(pack1)
    assert crc2 == _crc_over_pack_without_crc(pack2)

    # Numeric arrays equal
    for k in ("link_weights", "mc_policy_words", "cat_masks", "cpu_affinities", "dvfs_states"):
        a = pack1[k]
        b = pack2[k]
        assert np.array_equal(a, b)
    assert np.array_equal(pack1["noc_tables"]["weights"], pack2["noc_tables"]["weights"])


def test_link_weights_alias():
    H, W = 5, 7
    g, phi, grad, U, J, B, geom_meta = _build_inputs(H, W)
    cfg = PackerConfig(dvfs_levels=[0, 1, 2])

    pack = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg)
    assert np.allclose(pack["link_weights"], pack["noc_tables"]["weights"], atol=0.0)


def test_cat_policy_modes():
    H, W = 6, 6
    g, phi, grad, U, J, B, geom_meta = _build_inputs(H, W)

    cfg_uniform = PackerConfig(dvfs_levels=[0, 1, 2], cat_policy="uniform", cat_num_ways=8)
    cfg_weighted = PackerConfig(dvfs_levels=[0, 1, 2], cat_policy="u_weighted", cat_num_ways=8)

    pack_u = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg_uniform)
    pack_w = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg_weighted)

    cat_u = pack_u["cat_masks"]
    cat_w = pack_w["cat_masks"]

    # Uniform should enable floor(ways/2) bits exactly
    expected_bits = cfg_uniform.cat_num_ways // 2
    # count bits per element (portable popcount)
    popcount = np.vectorize(lambda z: bin(int(z)).count("1"))
    bits_u = popcount(cat_u.astype(int))
    assert np.all(bits_u == expected_bits)

    bits_w = popcount(cat_w.astype(int))
    # Weighted should not be strictly less enabled than uniform on average
    assert bits_w.mean() >= expected_bits - 1e-6
    # And at least one tile should have more enabled ways if U varies
    assert np.any(bits_w > bits_u) or np.any(bits_w < bits_u)  # allow variation either way


def test_dvfs_from_modes():
    H, W = 8, 4
    g, phi, grad, U, J, B, geom_meta = _build_inputs(H, W)
    levels = [0, 1, 2, 3]

    cfg_g = PackerConfig(dvfs_levels=levels, dvfs_from="grad_phi")
    cfg_u = PackerConfig(dvfs_levels=levels, dvfs_from="U")

    pack_g = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg_g)
    pack_u = make_reconfig_pack(g, phi, grad, U, J, B, geom_meta, cfg_u)

    s_g = pack_g["dvfs_states"]
    s_u = pack_u["dvfs_states"]

    # Different strategies should produce different mappings for nontrivial inputs
    assert s_g.shape == (H, W) and s_u.shape == (H, W)
    assert not np.array_equal(s_g, s_u)

    # Values are from allowed domain
    allowed = set(levels)
    assert all(int(v) in allowed for v in s_g.ravel())
    assert all(int(v) in allowed for v in s_u.ravel())