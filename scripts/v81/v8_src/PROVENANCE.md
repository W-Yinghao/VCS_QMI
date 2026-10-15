# Provenance of the read-only V8 / V7 source copies used by V8.1 fork 3 (D06 + checkpoint inference)

Copied 2026-10-15 from the v8 branch worktree /home/infres/yinwang/CS_QMI/ssl_pilot_v8 at commit b8ac563 (v8/utility-aligned-cs); the six files are
unchanged since the formal-unit commit cc6299f (git log cc6299f..b8ac563 -- <files> is empty).  Git blob SHA-1 (as in the v8 tree) and sha256 of the copies:

- scripts/v8/v8_train.py  blob 67f9dd5d6c4e5a9e1b02c8f14be5a55e20a9f83d  sha256 4464deb09ba13d8b
- scripts/v8/v8_core.py  blob 0542e843673a43a56a008ccd7c167cc5bdd8a3ec  sha256 5e38c0e06f427beb
- scripts/v8/v8_labels.py  blob 4e5c0ce8fa1a524fd12006028e55c4ff1995d317  sha256 d14386243494f9f9
- scripts/v7/v7_train.py  blob 11c7bfafb644e7b9add1a68d776c91352fccf262  sha256 0d8437f454b93a69
- scripts/v7/v7_common.py  blob 1080c8f915f8064101b838c68ad9d6fd5623b828  sha256 914f1b3a057cf777
- scripts/v7/v7_data.py  blob ee4986186735062758dfa299a2740e93f4433f0b  sha256 ec3f6f8514237b0f

Nothing in these copies is edited; v8_train.py's imports resolve to the sibling copies (HERE and HERE.parent/v7).  The frozen V8 evaluators
(scripts/v8/v8_eval.py, v8_confirm.py) are NOT copied: fork 3 never re-scores GenAI-Rank / GenAI-Bench / TIFA160.
