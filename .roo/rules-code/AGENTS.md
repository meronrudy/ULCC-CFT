# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Project Coding Rules (Non-Obvious Only):
- Maintain import-name symlinks locally and in containers before importing (ulcc_core ↔ ulcc-core, ulcc_ddg ↔ ulcc-ddg, ulcc_pggs ↔ pggs-toy).
- Always thread rng (numpy default_rng) through PGGS APIs; tests assume rng=np.random.default_rng(0). See [pggs_toy.sampler.sample_paths()](pggs-toy/sampler.py:7).
- Enforce domain guards in ulcc-core APIs: θ ∈ (0,1) only. See [theta_to_phi()](ulcc-core/coords.py:4), [fisher_metric_bernoulli()](ulcc-core/fisher.py:4).
- Preserve stub semantics: [parallel_transport_identity()](ulcc-ddg/transport.py:5)=1.0; [holonomy_loop()](ulcc_ddg/holonomy.py:7)=0.0 when transport is identity.
- Type discipline: scalars → float; vectors → np.ndarray float; keep alias “Gamma” (see [ulcc-core/dynamics.py](ulcc-core/dynamics.py:3)).
- Control pack handling: use [_to_native()](harness/control_loop.py:20) for numpy→JSON-native conversion before externalization.
- Edge/graph immutability: Edges are hashable (avoid in-place mutation). See [Edge](ulcc-ddg/metric_graph.py:5).