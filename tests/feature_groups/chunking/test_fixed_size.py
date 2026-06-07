"""Tests for FixedSizeChunker."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.chunking import FixedSizeChunker
from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker
from tests.feature_groups.chunking.text_chunking_test_base import TextChunkingTestBase


class TestFixedSizeChunker(TextChunkingTestBase):
    """Tests for FixedSizeChunker."""

    @property
    def chunker_class(self) -> Type[BaseChunker]:
        return FixedSizeChunker

    def test_chunk_short_text(self) -> None:
        """Short text should return single chunk."""
        chunks = FixedSizeChunker._chunk_text("Hello world", 100, 10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunk_long_text(self) -> None:
        """Long text should be split into multiple chunks."""
        text = "This is a test. " * 50  # ~800 chars
        chunks = FixedSizeChunker._chunk_text(text, 100, 10)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_chunk_overlap(self) -> None:
        """Chunks should have overlap."""
        text = "word " * 100  # 500 chars
        chunks = FixedSizeChunker._chunk_text(text, 100, 50)
        assert len(chunks) > 2

    @staticmethod
    def _leading_overlap(first: str, second: str) -> int:
        """Length of the longest suffix of ``first`` that is a prefix of ``second``."""
        for size in range(min(len(first), len(second)), 0, -1):
            if first[-size:] == second[:size]:
                return size
        return 0

    def test_overlap_stays_consistent_after_word_boundary_trim(self) -> None:
        """Word-boundary trims must not erode the requested overlap (issue #6).

        Varied word lengths force trims at different offsets within each chunk.
        Before the fix, ``start`` advanced by the full fixed step regardless of
        how far the trim pulled ``end`` back, so the real overlap drifted well
        below ``chunk_overlap``. The overlap should now stay at ``chunk_overlap``
        (give or take one character lost to whitespace stripping).
        """
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 10
        chunk_size, chunk_overlap = 80, 20
        chunks = FixedSizeChunker._chunk_text(text, chunk_size, chunk_overlap)
        assert len(chunks) > 2

        overlaps = [self._leading_overlap(a, b) for a, b in zip(chunks, chunks[1:])]
        assert all(overlap >= chunk_overlap - 1 for overlap in overlaps), overlaps
