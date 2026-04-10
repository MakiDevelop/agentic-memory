"""Memory lifecycle automation.

Provides automated maintenance operations:
- Auto-expire: remove memories past their TTL
- Auto-downgrade: lower importance of stale memories, mark as auto_downgraded
- Auto-compact: remove low-adoption memories based on usage metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from agentic_memory.models import ValidationStatus

if TYPE_CHECKING:
    from agentic_memory.memory import Memory


@dataclass
class LifecycleResult:
    """Outcome of a lifecycle operation."""

    expired_removed: int = 0
    stale_downgraded: int = 0
    low_adoption_removed: int = 0
    total_before: int = 0
    total_after: int = 0


class LifecycleManager:
    """Orchestrates automated memory lifecycle operations.

    Wraps a Memory instance and provides higher-level maintenance
    than the basic Memory.compact() method.
    """

    def __init__(self, memory: Memory):
        self._memory = memory

    def auto_expire(self) -> int:
        """Remove all memories that have exceeded their TTL.

        Returns number of memories removed.
        """
        removed = 0
        all_memories = self._memory._store.list_all(limit=None)
        for record in all_memories:
            if record.is_expired:
                self._memory._store.delete(record.id)
                removed += 1
        return removed

    def auto_downgrade_stale(self, importance_penalty: int = 1) -> int:
        """Lower importance of stale memories and mark them auto_downgraded.

        Re-validates all memories first. Those that become STALE are:
        - Marked as auto_downgraded = 1
        - Importance reduced by `importance_penalty` (floor 0)

        Returns number of memories downgraded.
        """
        downgraded = 0
        all_memories = self._memory._store.list_all(limit=None)
        for record in all_memories:
            # Skip already-downgraded memories
            if self._is_auto_downgraded(record.id):
                continue

            self._memory._validate_record(record)
            if record.validation_status == ValidationStatus.STALE:
                new_importance = max(0, record.importance - importance_penalty)
                self._memory._store._conn.execute(
                    "UPDATE memories SET importance = ?, auto_downgraded = 1, updated_at = ? WHERE id = ?",
                    (new_importance, datetime.now().isoformat(), record.id),
                )
                self._memory._store._conn.commit()
                downgraded += 1
        return downgraded

    def auto_compact_by_adoption(self, min_adoption_count: int = 0, min_age_days: int = 30) -> int:
        """Remove memories with low adoption that are older than min_age_days.

        A memory is removed if ALL of:
        - Its adoption count is <= min_adoption_count
        - It's older than min_age_days (by created_at)
        - It's not marked as critical importance (importance < 3)

        Returns number of memories removed.
        """
        removed = 0
        adoption_counts = self._memory._store.get_adoption_counts()
        all_memories = self._memory._store.list_all(limit=None)
        now = datetime.now()

        for record in all_memories:
            if record.importance >= 3:
                continue  # protect critical memories
            age_days = (now - record.created_at).total_seconds() / 86400
            if age_days < min_age_days:
                continue
            count = adoption_counts.get(record.id, 0)
            if count <= min_adoption_count:
                self._memory._store.delete(record.id)
                removed += 1
        return removed

    def run_all(
        self,
        *,
        min_adoption_count: int = 0,
        min_age_days: int = 30,
        importance_penalty: int = 1,
    ) -> LifecycleResult:
        """Run the full lifecycle pipeline: expire → downgrade → compact."""
        total_before = self._memory._store.count()

        expired = self.auto_expire()
        downgraded = self.auto_downgrade_stale(importance_penalty=importance_penalty)
        low_adoption = self.auto_compact_by_adoption(
            min_adoption_count=min_adoption_count,
            min_age_days=min_age_days,
        )

        total_after = self._memory._store.count()

        return LifecycleResult(
            expired_removed=expired,
            stale_downgraded=downgraded,
            low_adoption_removed=low_adoption,
            total_before=total_before,
            total_after=total_after,
        )

    def _is_auto_downgraded(self, memory_id: str) -> bool:
        """Check if a memory is already marked as auto_downgraded."""
        row = self._memory._store._conn.execute(
            "SELECT auto_downgraded FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return False
        return bool(row[0])
