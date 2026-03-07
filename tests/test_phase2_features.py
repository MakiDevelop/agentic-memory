"""Tests for Phase 2 features: conflict detection, compact, create_if_useful, search_context, eval_metrics."""

from datetime import datetime, timedelta

from agentic_memory import AddResult, CompactResult, EvalMetrics, ManualRef, Memory, MemoryKind


class TestConflictDetection:
    def test_no_conflict_for_unique_content(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("uses ruff for linting", evidence=ManualRef("docs"))
        record = mem.add("uses pytest for testing", evidence=ManualRef("docs"))
        assert record.conflict_ids == []
        mem.close()

    def test_conflict_detected_for_similar_topic(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("uses ruff for linting with line-length 120", evidence=ManualRef("docs"))
        record = mem.add("uses black for linting with line-length 88", evidence=ManualRef("other"))
        # FTS should find the first record as a candidate since they share "linting" and "line-length"
        assert len(record.conflict_ids) >= 1
        mem.close()

    def test_duplicate_not_marked_as_conflict(self, tmp_path):
        mem = Memory(str(tmp_path))
        r1 = mem.add("uses ruff for linting", evidence=ManualRef("docs"))
        r2 = mem.add("uses ruff for linting", evidence=ManualRef("other"))
        # Dedup returns existing, no conflict
        assert r1.id == r2.id
        mem.close()

    def test_detect_conflicts_method(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("project uses ruff for code linting", evidence=ManualRef("docs"))
        conflicts = mem.detect_conflicts("project uses black for code linting")
        assert len(conflicts) >= 1
        mem.close()

    def test_detect_conflicts_empty_when_no_match(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("uses ruff for linting", evidence=ManualRef("docs"))
        conflicts = mem.detect_conflicts("database uses postgres")
        assert len(conflicts) == 0
        mem.close()


class TestCompact:
    def test_compact_removes_expired(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("permanent fact", evidence=ManualRef("a"))
        record = mem.add("temp fact", evidence=ManualRef("b"), ttl_seconds=1)
        record.created_at = datetime.now() - timedelta(seconds=10)
        mem._store.save(record)

        result = mem.compact()
        assert isinstance(result, CompactResult)
        assert result.expired_removed == 1
        assert result.total_before == 2
        assert result.total_after == 1
        mem.close()

    def test_compact_no_expired(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("fact one", evidence=ManualRef("a"))
        mem.add("fact two", evidence=ManualRef("b"))

        result = mem.compact()
        assert result.expired_removed == 0
        assert result.total_before == 2
        assert result.total_after == 2
        mem.close()

    def test_compact_removes_all_expired(self, tmp_path):
        mem = Memory(str(tmp_path))
        for i in range(3):
            r = mem.add(f"temp fact {i}", evidence=ManualRef("x"), ttl_seconds=1, deduplicate=False)
            r.created_at = datetime.now() - timedelta(seconds=10)
            mem._store.save(r)

        result = mem.compact()
        assert result.expired_removed == 3
        assert result.total_after == 0
        mem.close()


class TestCreateIfUseful:
    def test_create_if_useful_success(self, tmp_path):
        mem = Memory(str(tmp_path))
        result = mem.create_if_useful("important rule", evidence=ManualRef("docs"), importance=2)
        assert result is not None
        assert isinstance(result, AddResult)
        assert result.record.importance == 2
        mem.close()

    def test_create_if_useful_rejected_low_importance(self, tmp_path):
        mem = Memory(str(tmp_path))
        result = mem.create_if_useful(
            "minor note", evidence=ManualRef("chat"), importance=0, min_importance=2
        )
        assert result is None
        mem.close()

    def test_create_if_useful_detects_duplicate(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("uses ruff", evidence=ManualRef("docs"))
        result = mem.create_if_useful("uses ruff", evidence=ManualRef("other"))
        assert result is not None
        assert result.was_duplicate is True
        mem.close()

    def test_create_if_useful_with_conflicts(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("project uses ruff for code linting", evidence=ManualRef("docs"))
        result = mem.create_if_useful("project uses black for code linting", evidence=ManualRef("other"))
        assert result is not None
        assert len(result.conflicts) >= 1
        mem.close()


class TestAddWithResult:
    def test_add_with_result_basic(self, tmp_path):
        mem = Memory(str(tmp_path))
        result = mem.add_with_result("new fact here", evidence=ManualRef("docs"))
        assert isinstance(result, AddResult)
        assert result.was_duplicate is False
        assert result.record.content == "new fact here"
        mem.close()

    def test_add_with_result_duplicate(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("existing fact", evidence=ManualRef("docs"))
        result = mem.add_with_result("existing fact", evidence=ManualRef("other"))
        assert result.was_duplicate is True
        mem.close()


class TestSearchContext:
    def test_search_context_returns_formatted_string(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("uses ruff for linting", evidence=ManualRef("docs"), kind="rule", importance=2)
        ctx = mem.search_context("linting")
        assert "ruff" in ctx
        assert "rule" in ctx
        assert "importance=2" in ctx
        mem.close()

    def test_search_context_no_results(self, tmp_path):
        mem = Memory(str(tmp_path))
        ctx = mem.search_context("nonexistent topic")
        assert "No memories found" in ctx
        mem.close()

    def test_search_context_with_kind_filter(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("ruff is the linter", evidence=ManualRef("docs"), kind="fact")
        mem.add("always run ruff before commit", evidence=ManualRef("rules"), kind="rule")
        ctx = mem.search_context("ruff", kind="rule")
        assert "always run" in ctx
        assert "Found 1 memories" in ctx
        mem.close()


class TestEvalMetrics:
    def test_eval_metrics_basic(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("fact one", evidence=ManualRef("a"))
        mem.add("fact two", evidence=ManualRef("b"))
        mem.query("fact")
        mem.query("two")

        metrics = mem.eval_metrics()
        assert isinstance(metrics, EvalMetrics)
        assert metrics.total_queries == 2
        assert metrics.total_memories == 2
        assert metrics.avg_latency_ms >= 0
        assert metrics.avg_results_per_query >= 0
        mem.close()

    def test_eval_metrics_empty(self, tmp_path):
        mem = Memory(str(tmp_path))
        metrics = mem.eval_metrics()
        assert metrics.total_queries == 0
        assert metrics.total_memories == 0
        assert metrics.avg_latency_ms == 0.0
        mem.close()

    def test_eval_metrics_counts_stale_and_expired(self, tmp_path):
        mem = Memory(str(tmp_path))
        mem.add("permanent", evidence=ManualRef("a"))
        r = mem.add("expiring", evidence=ManualRef("b"), ttl_seconds=1)
        r.created_at = datetime.now() - timedelta(seconds=10)
        mem._store.save(r)

        metrics = mem.eval_metrics()
        assert metrics.expired_count == 1
        mem.close()
