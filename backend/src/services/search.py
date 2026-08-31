"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
from hello_agents.tools import SearchTool
from hello_agents.tools import Tool
from hello_agents.tools import ToolParameter

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")

from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def _build_result_text(result: dict[str, Any]) -> str:
    """Build text used for semantic relevance calculation."""

    title = str(result.get("title") or "")
    snippet = str(
        result.get("snippet")
        or result.get("content")
        or result.get("text")
        or ""
    )

    return f"{title}\n{snippet}".strip()

def _calculate_relevance(
    query: str,
    results: list[dict[str, Any]],
RELEVANCE_THRESHOLD=0.3) -> list[dict[str, Any]]:
    """Calculate semantic relevance between query and search results."""

    if not results:
        return []
    # 一次性计算 query + 所有 result 的 embedding
    texts = [query] + [_build_result_text(result) for result in results]

    embeddings = _EMBEDDING_MODEL.encode(texts,normalize_embeddings=True,)

    query_embedding = embeddings[0]
    result_embeddings = embeddings[1:]

    # 因为已经 normalize_embeddings=True，
    # cosine similarity 就等价于向量点积
    scores = np.dot(result_embeddings, query_embedding)

    filtered_results = []

    for result, score in zip(results, scores):
        score = float(score)

        result = dict(result)
        result["relevance_score"] = score

        logger.info(
            "Search result relevance=%.4f title=%s",
            score,
            result.get("title"),)

        if score >= RELEVANCE_THRESHOLD:
            filtered_results.append(result)

    logger.info("Semantic relevance filtering: %d -> %d results, threshold=%.2f",
        len(results),
        len(filtered_results),
        RELEVANCE_THRESHOLD,)
    return filtered_results

def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)
    logger.info("dispatch_search query=%s", query)
    logger.info("dispatch_search backend=%s", search_api)
    try:
        raw_response = _GLOBAL_SEARCH_TOOL.run(
            {
                "input": query,
                "backend": search_api,
                "mode": "structured",
                "fetch_full_page": config.fetch_full_page,
                "max_results": 5,
                "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                "loop_count": loop_count,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    if results:
        results = _calculate_relevance(
            query=query,
            results=results,
        )

        # 把过滤后的结果重新放回 payload
        payload["results"] = results

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context


class EvidenceSearchTool(Tool):
    """供 Evidence Agent 自主调用的批量证据搜索工具。"""

    def __init__(self, config: Configuration):
        super().__init__(
            name="search",
            description=(
                "搜索互联网并返回与查询相关的网页证据。"
                "一次只提交一个具体的待核验事实或结论。"
            ),
        )
        self.config = config

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description=(
                    "需要核验的事实列表。"
                    "每个元素是一条独立的事实、数字、研究结论或具体说法。"
                ),
                required=True,
            )
        ]

    def run(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = (parameters.get("query") or "").strip()

        if not query:
            return {
                "error": "搜索关键词不能为空"
            }

        try:
            search_result, notices, answer_text, backend = dispatch_search(
                query,
                self.config,
                0,
            )

            _, context = prepare_research_context(
                search_result,
                answer_text,
                self.config,
            )

            return {
                "query": query,
                "backend": backend,
                "context": context,
                "notices": notices,
            }

        except Exception as e:
            logger.exception(
                "Evidence search failed for query=%s",
                query,
            )

            return {
                "query": query,
                "error": str(e),
            }