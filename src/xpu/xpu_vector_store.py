"""XPU vector store backed by PostgreSQL + pgvector.

Schema (table xpu_entries by default):
- id           TEXT PRIMARY KEY
- signals      JSONB  (applicability + regex + keywords + situation_triggers)
- advice_nl    JSONB
- atoms        JSONB
- embedding    vector(1536)
- telemetry    JSONB  (hits / successes / failures)
- created_at   TIMESTAMP
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np  # noqa: F401  (kept for downstream extensions)
import psycopg2
from psycopg2.extras import execute_values  # noqa: F401
from psycopg2.pool import ThreadedConnectionPool

from .xpu_adapter import XpuEntry, XpuContext
from ..logger import get_logger

logger = get_logger("xpu.vector_store")

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

# Table name override allows isolating experimental groups in the same DB
# (e.g. clean_warm vs clean_cold).
XPU_TABLE = os.environ.get("XPU_TABLE", "xpu_entries")


# ---------------------------------------------------------------------------
# Connection & schema
# ---------------------------------------------------------------------------

def get_db_connection_string() -> str:
    dns = os.environ.get("dns")
    if not dns:
        raise RuntimeError("missing required env var: dns (PostgreSQL connection string)")
    return dns


def create_xpu_table(conn, table_name: str = None) -> None:
    """Create the XPU table and IVFFlat index if absent."""
    tbl = table_name or XPU_TABLE
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
                id TEXT PRIMARY KEY,
                signals JSONB NOT NULL,
                advice_nl JSONB NOT NULL,
                atoms JSONB NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                telemetry JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {tbl}_embedding_idx
            ON {tbl}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        conn.commit()
        logger.info(f"XPU table {tbl} and index ready")


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def text_to_embedding(text: str, model: str = None) -> List[float]:
    """Generate an embedding via an OpenAI-compatible API.

    Config priority:
      1. EMBEDDING_API_KEY (+ EMBEDDING_BASE_URL / EMBEDDING_MODEL) -> dedicated service
      2. OPENAI_API_KEY (+ OPENAI_BASE_URL) -> fallback
    """
    import openai

    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL")

    if embedding_api_key:
        api_key = embedding_api_key
        base_url = embedding_base_url
        model = model or embedding_model or "text-embedding-3-small"
        logger.info(f"using embedding API: {base_url or 'default'}, model: {model}")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = model or "text-embedding-3-small"

        if not api_key:
            raise RuntimeError(
                "missing API key for embedding generation; "
                "set EMBEDDING_API_KEY or OPENAI_API_KEY"
            )

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Build searchable text for an XPU
# ---------------------------------------------------------------------------

def build_xpu_text(entry: XpuEntry) -> str:
    """Concat XPU fields into a searchable text. Keywords are repeated 3x to
    upweight them in embedding space.
    """
    parts = []

    ctx = entry.signals.get("applicability", {}) or {}
    if ctx.get("lang"):
        parts.append(f"Language: {ctx['lang']}")
    if ctx.get("tools"):
        parts.append(f"Tools: {', '.join(ctx['tools'])}")
    if ctx.get("python"):
        parts.append(f"Python versions: {', '.join(map(str, ctx['python']))}")
    if ctx.get("os"):
        parts.append(f"OS: {', '.join(ctx['os'])}")

    signals = entry.signals
    if signals.get("keywords"):
        keywords_str = ', '.join(signals['keywords'])
        parts.append(f"Keywords: {keywords_str}")
        parts.append(f"Error keywords: {keywords_str}")
        parts.append(f"Matching signals: {keywords_str}")
    if signals.get("regex"):
        parts.append(f"Error patterns: {', '.join(signals['regex'])}")
    if signals.get("situation_triggers"):
        parts.append(f"Situation: {'; '.join(signals['situation_triggers'])}")

    if entry.advice_nl:
        parts.append("Advice: " + " ".join(entry.advice_nl))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class XpuVectorStore:
    """PostgreSQL + pgvector backed XPU store with a thread-safe connection pool."""

    def __init__(self, connection_string: Optional[str] = None, table_name: Optional[str] = None):
        self.connection_string = connection_string or get_db_connection_string()
        self._table = table_name or XPU_TABLE
        self.pool = ThreadedConnectionPool(1, 5, self.connection_string)
        self._ensure_table()

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        try:
            create_xpu_table(conn, self._table)
        finally:
            self._put_conn(conn)

    # -----------------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------------

    def upsert_entry(self, entry: XpuEntry, embedding: List[float]) -> None:
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"embedding dim mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}")

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # pgvector literal: '[0.1,0.2,...]'
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

                cur.execute(f"""
                    INSERT INTO {self._table} (id, signals, advice_nl, atoms, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        signals = EXCLUDED.signals,
                        advice_nl = EXCLUDED.advice_nl,
                        atoms = EXCLUDED.atoms,
                        embedding = EXCLUDED.embedding;
                """, (
                    entry.id,
                    json.dumps(entry.signals),
                    json.dumps(entry.advice_nl),
                    json.dumps([{"name": a.name, "args": a.args} for a in entry.atoms]),
                    embedding_str,
                ))
                conn.commit()
        finally:
            self._put_conn(conn)

    # -----------------------------------------------------------------------
    # Vector similarity search with composite scoring
    # -----------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        ctx: Optional[XpuContext] = None,
        k: int = 3,
        min_similarity: float = 0.45,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine-similarity search with telemetry-weighted ranking.

        composite_score = similarity * (1 + success_rate) * tier_boost
        success_rate    = successes / max(hits, 1)

        Tier boost (computed live from telemetry):
          hits >= 5 and success_rate >= 0.6   -> 1.5  (golden)
          hits >= 5 and success_rate <  0.3   -> 0.6  (cold)
          otherwise                           -> 1.0  (normal)

        When env RETRIEVER_MODE=direct, ranking falls back to pure similarity.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(f"query embedding dim mismatch: expected {EMBEDDING_DIM}, got {len(query_embedding)}")

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Bump probes so small datasets don't miss matches via the IVFFlat index.
                cur.execute("SET ivfflat.probes = 10")

                where_clauses = []
                where_params = []

                if ctx:
                    if ctx.lang:
                        if isinstance(ctx.lang, (list, tuple, set)):
                            where_clauses.append("signals->'applicability'->>'lang' = ANY(%s)")
                            where_params.append(list(ctx.lang))
                        else:
                            where_clauses.append("signals->'applicability'->>'lang' = %s")
                            where_params.append(ctx.lang)
                    if ctx.python:
                        py_list = ctx.python if isinstance(ctx.python, (list, tuple, set)) else [ctx.python]
                        py_conditions = []
                        for py_ver in py_list:
                            py_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(signals->'applicability'->'python') AS v WHERE v LIKE %s)")
                            where_params.append(f"{py_ver}%")
                        if py_conditions:
                            where_clauses.append(f"({' OR '.join(py_conditions)})")
                    if ctx.tools:
                        tool_conditions = []
                        for tool in ctx.tools:
                            tool_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(signals->'applicability'->'tools') AS t WHERE t = %s)")
                            where_params.append(tool)
                        if tool_conditions:
                            where_clauses.append(f"({' OR '.join(tool_conditions)})")

                # Negative-feedback filtering is handled by RetrieverAgent (soft filter)
                # rather than at the SQL layer.

                if exclude_ids:
                    where_clauses.append("id != ALL(%s)")
                    where_params.append(list(exclude_ids))

                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

                embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

                tier_boost_expr = """
                    CASE
                        WHEN COALESCE((telemetry->>'hits')::int, 0) >= 5
                             AND COALESCE((telemetry->>'successes')::float, 0)
                                 / GREATEST(COALESCE((telemetry->>'hits')::int, 0), 1) >= 0.6
                        THEN 1.5
                        WHEN COALESCE((telemetry->>'hits')::int, 0) >= 5
                             AND COALESCE((telemetry->>'successes')::float, 0)
                                 / GREATEST(COALESCE((telemetry->>'hits')::int, 0), 1) < 0.3
                        THEN 0.6
                        ELSE 1.0
                    END
                """

                direct_mode = os.environ.get("RETRIEVER_MODE") == "direct"

                if direct_mode:
                    query = f"""
                        SELECT
                            id, signals, advice_nl, atoms,
                            1 - (embedding <=> %s::vector) AS similarity,
                            telemetry,
                            1 - (embedding <=> %s::vector) AS composite_score
                        FROM {self._table}
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                        {where_sql}
                        ORDER BY similarity DESC
                        LIMIT %s;
                    """
                    params = (
                        [embedding_str, embedding_str, embedding_str, min_similarity]
                        + where_params
                        + [k]
                    )
                else:
                    query = f"""
                        SELECT
                            id,
                            signals,
                            advice_nl,
                            atoms,
                            1 - (embedding <=> %s::vector) AS similarity,
                            telemetry,
                            (1 - (embedding <=> %s::vector))
                                * (1.0 + COALESCE((telemetry->>'successes')::float, 0)
                                   / GREATEST(COALESCE((telemetry->>'hits')::int, 0), 1))
                                * ({tier_boost_expr})
                            AS composite_score
                        FROM {self._table}
                        WHERE 1 - (embedding <=> %s::vector) >= %s
                        {where_sql}
                        ORDER BY composite_score DESC
                        LIMIT %s;
                    """
                    params = (
                        [embedding_str, embedding_str]
                        + [embedding_str, min_similarity]
                        + where_params
                        + [k]
                    )

                cur.execute(query, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    telemetry = row[5] or {}
                    hits = int(telemetry.get("hits", 0))
                    successes = float(telemetry.get("successes", 0))
                    success_rate = successes / max(hits, 1)
                    if hits >= 5 and success_rate >= 0.6:
                        tier = "golden"
                    elif hits >= 5 and success_rate < 0.3:
                        tier = "cold"
                    else:
                        tier = "normal"

                    results.append({
                        "id": row[0],
                        "signals": row[1],
                        "advice_nl": row[2],
                        "atoms": row[3],
                        "similarity": float(row[4]),
                        "telemetry": telemetry,
                        "composite_score": float(row[6]),
                        "tier": tier,
                    })

                return results
        finally:
            self._put_conn(conn)

    def get_entry(self, xpu_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, signals, advice_nl, atoms
                    FROM {self._table}
                    WHERE id = %s;
                """, (xpu_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "signals": row[1],
                    "advice_nl": row[2],
                    "atoms": row[3],
                }
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        if self.pool and not self.pool.closed:
            self.pool.closeall()

    # -----------------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------------

    _TELEMETRY_FIELDS = {"hits", "successes", "failures"}

    def increment_telemetry(self, xpu_ids: List[str], field: str):
        """Atomic +1 on telemetry.<field> for each id in xpu_ids.

        Skipped when env FREEZE_TELEMETRY=1 (used to protect telemetry during runs).
        """
        if os.environ.get("FREEZE_TELEMETRY") == "1":
            logger.debug(f"FREEZE_TELEMETRY=1, skip telemetry update: {field}")
            return
        if field not in self._TELEMETRY_FIELDS:
            raise ValueError(f"illegal telemetry field: {field}, allowed: {self._TELEMETRY_FIELDS}")
        if not xpu_ids: return
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                sql = f"""
                    UPDATE {self._table}
                    SET telemetry = jsonb_set(
                        COALESCE(telemetry, '{{}}'::jsonb),
                        '{{{field}}}',
                        (COALESCE(telemetry->>'{field}', '0')::int + 1)::text::jsonb
                    )
                    WHERE id = ANY(%s);
                """
                cur.execute(sql, (xpu_ids,))
                conn.commit()
        except Exception as e:
            logger.error(f"telemetry update ({field}) failed: {e}")
        finally:
            self._put_conn(conn)

    def update_advice(self, xpu_id: str, new_advice: List[str]) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self._table} SET advice_nl = %s WHERE id = %s;",
                    (json.dumps(new_advice), xpu_id),
                )
                conn.commit()
                logger.info(f"updated advice_nl of '{xpu_id}' ({len(new_advice)} items)")
        except Exception as e:
            logger.error(f"failed to update advice_nl of '{xpu_id}': {e}")
        finally:
            self._put_conn(conn)

    def update_telemetry_scores(self, updates: Dict[str, float], field: str = 'hits'):
        """Batch float-increment a telemetry field; supports weighted attribution."""
        if os.environ.get("FREEZE_TELEMETRY") == "1":
            logger.debug(f"FREEZE_TELEMETRY=1, skip batch telemetry update: {field}")
            return
        if field not in self._TELEMETRY_FIELDS:
            raise ValueError(f"illegal telemetry field: {field}, allowed: {self._TELEMETRY_FIELDS}")
        if not updates: return
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                for xpu_id, score in updates.items():
                    sql = f"""
                        UPDATE {self._table}
                        SET telemetry = jsonb_set(
                            COALESCE(telemetry, '{{}}'::jsonb),
                            '{{{field}}}',
                            to_jsonb(COALESCE((telemetry->>'{field}')::numeric, 0) + %s)
                        )
                        WHERE id = %s;
                    """
                    cur.execute(sql, (score, xpu_id))
                conn.commit()
        except Exception as e:
            logger.error(f"batch telemetry score update failed: {e}")
        finally:
            self._put_conn(conn)
