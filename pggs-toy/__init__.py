try:
    from . import algebra, hypergraph, action, guide, sampler
except Exception:
    # Tolerate collection/import when directory name has a hyphen and isn't treated as a package.
    # The real imports are via the symlinked package ulcc_pggs/.
    pass
