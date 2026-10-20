"""V10 EPFL RGB-NIR activation (owner decision 2026-10-17 section 6.2, decision section 10 step 5).

audit     gates 1-4 (CPU, numpy / stdlib / PIL; no torch, no method, no registration)
guard     the EPFL data gate under an explicit, hash-pinned activation record (stdlib only)
host      the EPFL registration host for the V10 methods (torch; imported only by GPU / CPU-torch entry points)
geometry  EPFL warp, pixel-unit and effective-support checks (torch)

Importing this package alone imports nothing scientific and reads no data.
"""
