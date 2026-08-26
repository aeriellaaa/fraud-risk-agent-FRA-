"""
Audit logging.

Writes to two places:
  1. The in-memory store (for fast reads via the /audit-log endpoint)
  2. audit_log.jsonl on disk, opened in append mode only -- never
     rewritten, never truncated. This is what makes "append-only" a
     real property of the system rather than just a description of the
     in-memory list, which would vanish on restart.

audit_log.jsonl is gitignored deliberately -- it's runtime data, not
source code, and it will contain per-transaction detail we do not want
committed to version control.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from app.models import AuditEntry
from app.storage import store

AUDIT_FILE = Path("audit_log.jsonl")


def log_stage(transaction_id: str, stage: str, actor: str, data: dict) -> AuditEntry:
    entry = AuditEntry(
        transaction_id=transaction_id,
        timestamp=datetime.now(timezone.utc),
        stage=stage,
        actor=actor,
        data=data,
    )

    # In-memory, for fast API reads
    store.append_audit(entry)

    # On-disk, append-only. "a" mode never overwrites existing lines.
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")

    return entry


def read_audit_log(transaction_id: str | None = None) -> list[AuditEntry]:
    return store.get_audit_log(transaction_id)
