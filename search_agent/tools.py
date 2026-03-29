import os
import json
import math
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_fixed
load_dotenv()
from model import embeddings

_milvus_client = None  # 刚启动时，连接是空的
collection_name = "text_search_1"
_bm25_index = None
_bm25_corpus: List[str] = []
_bm25_tokenized: List[List[str]] = []

def get_milvus_client():
    """获取数据库连接（如果没连过，就现连；如果连过了，就直接用）"""
    global _milvus_client
    if _milvus_client is None:
        print("\n[🔌 系统底座] 首次触发内网搜索，正在与 Milvus 建立连接...")
        _milvus_client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
    return _milvus_client

def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)

def _load_corpus_from_milvus(refresh: bool = False, batch_size: int = 500) -> None:
    global _bm25_index, _bm25_corpus, _bm25_tokenized
    if _bm25_index is not None and not refresh:
        return

    client = get_milvus_client()
    all_texts: List[str] = []
    offset = 0
    while True:
        try:
            batch = client.query(
                collection_name=collection_name,
                expr="id >= 0",
                output_fields=["text"],
                limit=batch_size,
                offset=offset
            )
        except Exception as e:
            raise RuntimeError(f"Milvus 查询失败: {str(e)}")

        if not batch:
            break

        for row in batch:
            text = row.get("text") if isinstance(row, dict) else None
            if text:
                all_texts.append(text)

        if len(batch) < batch_size:
            break
        offset += batch_size

    _bm25_corpus = all_texts
    _bm25_tokenized = [_simple_tokenize(t) for t in _bm25_corpus]
    if _bm25_tokenized:
        _bm25_index = BM25Okapi(_bm25_tokenized)
    else:
        _bm25_index = None

def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return {k: 1.0 for k in score_map}
    return {k: (v - v_min) / (v_max - v_min) for k, v in score_map.items()}

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# 创建两个工具(tool)
# 1.内部知识库检索工具
@tool
def search_internal_knowledge(
    query: str = "",
    top_k: int = 5,
    bm25_top_k: int = 8,
    embed_top_k: int = 8,
    weight_bm25: float = 0.4,
    weight_embed: float = 0.6,
    refresh_bm25: bool = False,
    **kwargs
) -> str:
    """
    搜索内部知识库（BM25 + Embedding 混合检索 + 轻量 reranker）。
    对于用户的问题，优先使用此工具搜索信息。
    """
    if not query or not query.strip():
        query = (kwargs.get("input") or kwargs.get("question") or kwargs.get("text") or "").strip()
    if not query or not query.strip():
        return ""
    print(f"\n[📚 查阅内网] 正在知识库中检索: '{query}'")
    try:
        client = get_milvus_client()

        # Embedding 召回
        query_vector = embeddings.embed_query(query)
        embed_res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=embed_top_k,
            output_fields=["text"]
        )
        embed_docs: Dict[str, float] = {}
        for hit in embed_res[0]:
            text = hit["entity"].get("text")
            distance = hit.get("distance", 0.0)
            if text:
                embed_docs[text] = 1.0 / (1.0 + float(distance))

        # BM25 召回
        _load_corpus_from_milvus(refresh=refresh_bm25)
        bm25_docs: Dict[str, float] = {}
        if _bm25_index is not None:
            tokens = _simple_tokenize(query)
            bm25_scores = _bm25_index.get_scores(tokens)
            if bm25_scores is not None and len(bm25_scores) == len(_bm25_corpus):
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:bm25_top_k]
                for i in top_indices:
                    bm25_docs[_bm25_corpus[i]] = float(bm25_scores[i])

        # 归一化并合并
        embed_norm = _normalize_scores(embed_docs)
        bm25_norm = _normalize_scores(bm25_docs)
        merged: Dict[str, Dict[str, float]] = {}

        for text, score in embed_norm.items():
            merged.setdefault(text, {})["embed"] = score

        for text, score in bm25_norm.items():
            merged.setdefault(text, {})["bm25"] = score

        if not merged:
            return "内部知识库中未找到相关内容。"

        # 计算混合分数
        hybrid_scores: Dict[str, float] = {}
        for text, parts in merged.items():
            e = parts.get("embed", 0.0)
            b = parts.get("bm25", 0.0)
            hybrid_scores[text] = weight_embed * e + weight_bm25 * b

        # 轻量 reranker：向量相似度再次排序
        candidates = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_texts = [c[0] for c in candidates[: max(top_k * 3, 10)]]
        if candidate_texts:
            cand_vectors = embeddings.embed_documents(candidate_texts)
            rerank_scores: Dict[str, float] = {}
            for text, vec in zip(candidate_texts, cand_vectors):
                rerank_scores[text] = _cosine_similarity(query_vector, vec)

            rerank_norm = _normalize_scores(rerank_scores)
            final_scores: List[Tuple[str, float]] = []
            for text in candidate_texts:
                final_scores.append((text, rerank_norm.get(text, 0.0)))
            final_sorted = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_k]
        else:
            final_sorted = candidates[:top_k]

        results = []
        for idx, (text, score) in enumerate(final_sorted, start=1):
            origin = "bm25+embedding" if text in bm25_docs and text in embed_docs else (
                "bm25" if text in bm25_docs else "embedding"
            )
            results.append({
                "rank": idx,
                "text": text,
                "score": round(float(score), 6),
                "source": origin
            })

        return "内部知识库检索结果：" + json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"知识库检索失败，系统异常。报错：{str(e)}"

# 2.外部实时网络搜索工具
ddg_search = DuckDuckGoSearchRun()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def robust_web_search(query: str) -> str:
    return ddg_search.run(query)

@tool
def search_internet(query: str = "", **kwargs) -> str:
    """
    进行外网实时搜索。
    当用户询问最新的新闻、公众人物动态、当前股市、汇率或常识性外部问题时，使用此工具。
    """
    if not query or not query.strip():
        query = (kwargs.get("input") or kwargs.get("question") or kwargs.get("text") or "").strip()
    if not query or not query.strip():
        return ""
    print(f"\n[🌐 呼叫外网] 正在 Google/Bing 检索: '{query}'")
    try:
        result = robust_web_search(query)
        result = result.strip()     # 结果做简单清洗，避免把原始网页过多信息带入对话
        return f"外网检索结果：\n{result}"
    except Exception as e:
        print("   [🛡️ 工具降级] 网络搜索彻底挂了...")
        return "外部网络搜索当前不可用，请告知用户网络超时。"