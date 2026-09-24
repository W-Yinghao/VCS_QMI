"""VCS-QMI CIFAR-10 SSL pilot trainer (server integration of the collaborator reference).

The mathematical objective lives in ``reference/ssl_core.py`` and is imported, never re-implemented.
Everything here is engineering plumbing: data contract, models, schedule, logging, checkpoints, evaluation.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "vcs_ssl_agent_1.0"
