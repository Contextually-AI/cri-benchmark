"""SQLite-backed disk cache for LangChain LLM responses.

Implements :class:`langchain_core.caches.BaseCache` so it can be assigned
to the ``cache`` field of any :class:`BaseChatModel`.  Identical prompts
(same text + same model/temperature/max_tokens) return the stored response
instead of making a new API call.

Usage::

    from cri.utils.llm_cache import SQLiteDiskCache

    cache = SQLiteDiskCache(db_path=".cri_cache/llm_cache.sqlite")
    llm = ChatAnthropic(...)
    llm.cache = cache          # all ainvoke() calls now check the cache
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.caches import BaseCache
from langchain_core.outputs import ChatGeneration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.outputs import Generation

# Type expected by BaseCache.update / returned by BaseCache.lookup.
_RETURN_VAL = "Sequence[Generation]"


def _cache_key(prompt: str, llm_string: str) -> str:
    """SHA-256 hash of (prompt, llm_string) used as the DB primary key."""
    raw = prompt + "||" + llm_string
    return hashlib.sha256(raw.encode()).hexdigest()


class SQLiteDiskCache(BaseCache):
    """Persistent LLM response cache backed by a local SQLite database.

    Args:
        db_path: Path to the SQLite file.  Parent directories are created
            automatically if they don't exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key        TEXT PRIMARY KEY,
                response   TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- BaseCache interface ---------------------------------------------------

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Return cached generations for the given prompt+llm pair, or *None*."""
        key = _cache_key(prompt, llm_string)
        with self._lock:
            row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return _deserialize(row[0])

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: Sequence[Generation],
    ) -> None:
        """Store generations in the cache."""
        key = _cache_key(prompt, llm_string)
        blob = _serialize(return_val)
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response, created_at) VALUES (?, ?, ?)",
                (key, blob, now),
            )
            self._conn.commit()

    def clear(self, **kwargs: object) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()


# -- Serialisation helpers -----------------------------------------------------


def _serialize(generations: Sequence[Generation]) -> str:
    """Serialise a list of Generation objects to a JSON string."""
    return json.dumps([_gen_to_dict(g) for g in generations])


def _deserialize(blob: str) -> list[Generation]:
    """Reconstruct Generation objects from a JSON string."""
    return [_dict_to_gen(d) for d in json.loads(blob)]


def _gen_to_dict(g: Generation) -> dict[str, object]:
    """Convert a Generation (or ChatGeneration) to a plain dict."""
    d: dict[str, object] = {"text": g.text, "generation_info": g.generation_info}
    if isinstance(g, ChatGeneration):
        d["type"] = "chat"
        d["message"] = {
            "content": g.message.content,
            "type": g.message.type,
        }
    return d


def _dict_to_gen(d: dict[str, object]) -> Generation:
    """Reconstruct a Generation (or ChatGeneration) from a dict."""
    if d.get("type") == "chat":
        from langchain_core.messages import AIMessage

        msg_data = d["message"]
        assert isinstance(msg_data, dict)
        content = msg_data.get("content", "")
        assert isinstance(content, (str, list))
        return ChatGeneration(
            text=str(d["text"]),
            generation_info=d.get("generation_info"),  # type: ignore[arg-type]
            message=AIMessage(content=content),
        )
    from langchain_core.outputs import Generation

    return Generation(text=str(d["text"]), generation_info=d.get("generation_info"))  # type: ignore[arg-type]
