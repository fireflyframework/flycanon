# Copyright 2026 Firefly Software Foundation. SPDX-License-Identifier: Apache-2.0
"""Admitted source membership precedes each ranked candidate window."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from flycanon.core.services.retrieval.corpus_factory import CorpusContext
from flycanon.core.services.retrieval.fusion import ChunkHit, StoredChunk
from flycanon.core.services.retrieval.retrieval_service import Hit, RetrievalFilters, RetrievalService


class RankedCorpus:
    def __init__(self, rows, enabled=True):
        self.rows = rows
        self.enabled = enabled
        self.reads = 0

    async def bm25_search(self, query, *, top_k, source_ids=None, **scope):
        self.reads += 1
        rows = [
            r
            for r in self.rows
            if r.metadata.get("tenant_id", "forge") == scope["tenant_id"]
            and r.metadata.get("workspace_id", "current") == scope["workspace_id"]
            and (source_ids is None or r.doc_id in source_ids)
        ]
        return (
            [ChunkHit(chunk_id=r.chunk_id, doc_id=r.doc_id, content=r.content, score=1) for r in rows[:top_k]]
            if self.enabled
            else []
        )

    async def get_chunks(self, ids, **scope):
        return [r for r in self.rows if r.chunk_id in ids]


class RankedVectors:
    def __init__(self, rows, enabled=True):
        self.rows = rows
        self.enabled = enabled
        self.reads = 0

    async def search(self, vector, *, top_k, filters=None, **scope):
        self.reads += 1
        rows = [
            r
            for r in self.rows
            if r.metadata.get("tenant_id", "forge") == scope["tenant_id"]
            and r.metadata.get("workspace_id", "current") == scope["workspace_id"]
        ]
        for f in filters or []:
            assert f.field == "source_id" and f.operator == "in"
            rows = [r for r in rows if r.doc_id in f.value]
        return (
            [SimpleNamespace(document=SimpleNamespace(id=r.chunk_id)) for r in rows[:top_k]]
            if self.enabled
            else []
        )


def service(channel):
    rows = [
        StoredChunk(
            "foreign",
            "admitted",
            "foreign",
            0,
            "foreign admitted source",
            metadata={"tenant_id": "other", "workspace_id": "foreign"},
        )
    ]
    rows += [
        StoredChunk(str(i), "outside-" + str(i), "other", 0, "distracting higher ranked content")
        for i in range(60)
    ]
    rows += [StoredChunk("wanted-chunk", "admitted", "current", 0, "official admitted API content")]
    corpus = RankedCorpus(rows, channel != "vector")
    vectors = RankedVectors(rows, channel != "bm25")
    embeddings = MagicMock()
    embeddings.embed_one = AsyncMock(return_value=[0.1, 0.2, 0.3])
    svc = RetrievalService(
        context=CorpusContext(corpus=corpus, vector_store=vectors, backend="test"),
        embeddings=embeddings,
        source_repository=MagicMock(),
        chunk_repository=MagicMock(),
        knowledge_repository=MagicMock(),
        default_top_k=5,
        default_per_query_k=20,
        rrf_k=60,
    )

    async def hydrate(hits):
        return [
            Hit(
                chunk_id=h.chunk_id,
                source_id=h.doc_id,
                content=h.content,
                score=h.score,
                knowledge_item_id=None,
                knowledge_version=None,
                bm25_rank=None,
                vector_rank=None,
                metadata={},
            )
            for h in hits
        ]

    svc._hydrate = hydrate
    return svc, corpus, vectors


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["bm25", "vector", "hybrid"])
async def test_source_filter_precedes_bounded_ranking_in_each_channel(channel):
    svc, _, _ = service(channel)
    result = await svc.search(
        query="official API",
        tenant_id="forge",
        workspace_id="current",
        top_k=1,
        per_query_k=2,
        filters=RetrievalFilters(source_ids=["admitted"]),
    )
    assert [h.source_id for h in result.hits] == ["admitted"]
    assert result.hits[0].content == "official admitted API content"


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [None, []])
async def test_none_and_empty_list_preserve_unfiltered_source_selector_compatibility(selector):
    svc, corpus, vectors = service("hybrid")
    result = await svc.search(
        query="official API",
        tenant_id="forge",
        workspace_id="current",
        top_k=1,
        filters=RetrievalFilters(source_ids=selector),
    )
    assert [h.source_id for h in result.hits] == ["outside-0"]
    assert corpus.reads == vectors.reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["bm25", "vector", "hybrid"])
async def test_nonempty_selector_with_no_chunks_does_not_fall_back_to_global_candidates(channel):
    svc, _, _ = service(channel)
    result = await svc.search(
        query="official API",
        tenant_id="forge",
        workspace_id="current",
        filters=RetrievalFilters(source_ids=["has-no-chunks"]),
    )
    assert result.hits == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tenant", "workspace", "content"),
    [
        ("forge", "current", "official admitted API content"),
        ("other", "foreign", "foreign admitted source"),
        ("forge", "foreign", None),
    ],
)
async def test_source_membership_intersects_tenant_and_workspace_before_ranking(tenant, workspace, content):
    svc, _, _ = service("hybrid")
    result = await svc.search(
        query="official API",
        tenant_id=tenant,
        workspace_id=workspace,
        top_k=1,
        per_query_k=2,
        filters=RetrievalFilters(source_ids=["admitted"]),
    )
    assert [h.content for h in result.hits] == ([] if content is None else [content])
