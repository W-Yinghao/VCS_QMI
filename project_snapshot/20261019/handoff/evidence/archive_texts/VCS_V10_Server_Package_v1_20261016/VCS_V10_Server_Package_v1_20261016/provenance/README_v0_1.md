# V10 Design v0.1

Read `V10_DESIGN_CN.md` first. This is a design/proof-of-concept reference, not an approved full server-run protocol.

Files: design note, CPU algebra/gradient reference, executed check results, source ledger.

Run: `OPENBLAS_NUM_THREADS=1 python v10_reference.py`

Dependencies: Python 3.10+, NumPy, SciPy, PyTorch. Tested locally with NumPy 2.3.5, SciPy 1.17.0 and PyTorch 2.10.0+cpu.

The code accepts numerical features only, performs no external downloads and reads no research data. It is not the production HVP solver, raw-image trainer, or registration integration.

V8 remains closed; V9 continues under its current protocols. No GitHub write or server submission was made.
