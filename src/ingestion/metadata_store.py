"""
src/ingestion/metadata_store.py

SQLite-backed store for raw pet metadata.
Tracks sync state, active/inactive status, and full raw JSON
so we can re-embed without re-fetching from the API.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parents[2] / "data" / "pets.db"

CREATE_PETS_TABLE = """
CREATE TABLE IF NOT EXISTS pets (
    rescuegroups_id TEXT PRIMARY KEY,
    org_id          TEXT,
    org_name        TEXT,
    name            TEXT,
    species         TEXT,
    breed           TEXT,
    age_group       TEXT,
    size            TEXT,
    sex             TEXT,
    energy_level    TEXT,
    activity_level  TEXT,
    good_with_kids  INTEGER,
    good_with_dogs  INTEGER,
    good_with_cats  INTEGER,
    is_seniors_ok   INTEGER,
    requires_yard   INTEGER,
    house_trained   INTEGER DEFAULT 0,
    is_altered      INTEGER DEFAULT 0,
    special_needs   INTEGER DEFAULT 0,
    shots_current   INTEGER DEFAULT 0,
    microchipped    INTEGER DEFAULT 0,
    city            TEXT,
    state           TEXT,
    postcode        TEXT,
    status          TEXT DEFAULT 'adoptable',
    photo_url       TEXT,
    adoption_url    TEXT,
    raw_json        TEXT,
    first_seen_at   TEXT,
    last_seen_at    TEXT,
    is_active       INTEGER DEFAULT 1
);
"""

CREATE_SYNC_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT,
    finished_at     TEXT,
    location        TEXT,
    animals_fetched INTEGER DEFAULT 0,
    animals_added   INTEGER DEFAULT 0,
    animals_updated INTEGER DEFAULT 0,
    animals_removed INTEGER DEFAULT 0,
    error           TEXT
);
"""


def _b(value: Optional[bool]) -> Optional[int]:
    return None if value is None else int(value)


class MetadataStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(CREATE_PETS_TABLE)
            self.conn.execute(CREATE_SYNC_LOG_TABLE)

    def upsert_pet(self, metadata: Dict[str, Any], raw_json: Dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        pid = metadata["rescuegroups_id"]

        existing = self.conn.execute(
            "SELECT rescuegroups_id, first_seen_at FROM pets WHERE rescuegroups_id = ?",
            (pid,)
        ).fetchone()

        row = {
            "rescuegroups_id": pid,
            "org_id":          metadata.get("org_id", ""),
            "org_name":        metadata.get("org_name", ""),
            "name":            metadata.get("name", ""),
            "species":         metadata.get("species", ""),
            "breed":           metadata.get("breed", ""),
            "age_group":       metadata.get("age_group", ""),
            "size":            metadata.get("size", ""),
            "sex":             metadata.get("sex", ""),
            "energy_level":    metadata.get("energy_level", ""),
            "activity_level":  metadata.get("activity_level", ""),
            "good_with_kids":  _b(metadata.get("good_with_kids")),
            "good_with_dogs":  _b(metadata.get("good_with_dogs")),
            "good_with_cats":  _b(metadata.get("good_with_cats")),
            "is_seniors_ok":   _b(metadata.get("is_seniors_ok")),
            "requires_yard":   _b(metadata.get("requires_yard")),
            "house_trained":   _b(metadata.get("house_trained")),
            "is_altered":      _b(metadata.get("is_altered")),
            "special_needs":   _b(metadata.get("special_needs")),
            "shots_current":   _b(metadata.get("shots_current")),
            "microchipped":    _b(metadata.get("microchipped")),
            "city":            metadata.get("city", ""),
            "state":           metadata.get("state", ""),
            "postcode":        metadata.get("postcode", ""),
            "status":          metadata.get("status", "adoptable"),
            "photo_url":       metadata.get("photo_url", ""),
            "adoption_url":    metadata.get("adoption_url", ""),
            "raw_json":        json.dumps(raw_json),
            "first_seen_at":   existing["first_seen_at"] if existing else now,
            "last_seen_at":    now,
            "is_active":       1,
        }

        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO pets
                  (rescuegroups_id, org_id, org_name, name, species, breed,
                   age_group, size, sex, energy_level, activity_level,
                   good_with_kids, good_with_dogs, good_with_cats, is_seniors_ok,
                   requires_yard, house_trained, is_altered, special_needs,
                   shots_current, microchipped, city, state, postcode,
                   status, photo_url, adoption_url, raw_json,
                   first_seen_at, last_seen_at, is_active)
                VALUES
                  (:rescuegroups_id, :org_id, :org_name, :name, :species, :breed,
                   :age_group, :size, :sex, :energy_level, :activity_level,
                   :good_with_kids, :good_with_dogs, :good_with_cats, :is_seniors_ok,
                   :requires_yard, :house_trained, :is_altered, :special_needs,
                   :shots_current, :microchipped, :city, :state, :postcode,
                   :status, :photo_url, :adoption_url, :raw_json,
                   :first_seen_at, :last_seen_at, :is_active)
            """, row)

        return "updated" if existing else "added"

    def mark_inactive(self, rescuegroups_ids: List[str]) -> int:
        if not rescuegroups_ids:
            return 0
        placeholders = ",".join("?" * len(rescuegroups_ids))
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            cursor = self.conn.execute(
                f"UPDATE pets SET is_active=0, last_seen_at=? WHERE rescuegroups_id IN ({placeholders})",
                [now, *rescuegroups_ids],
            )
        return cursor.rowcount

    def get_active_ids(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT rescuegroups_id FROM pets WHERE is_active=1"
        ).fetchall()
        return [r["rescuegroups_id"] for r in rows]

    def get_pet(self, rescuegroups_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM pets WHERE rescuegroups_id=?", (rescuegroups_id,)
        ).fetchone()
        return dict(row) if row else None

    def start_sync(self, location: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO sync_log (started_at, location) VALUES (?, ?)", (now, location)
            )
        return cursor.lastrowid

    def finish_sync(self, log_id: int, fetched: int, added: int,
                    updated: int, removed: int, error: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("""
                UPDATE sync_log
                SET finished_at=?, animals_fetched=?, animals_added=?,
                    animals_updated=?, animals_removed=?, error=?
                WHERE id=?
            """, (now, fetched, added, updated, removed, error, log_id))

    def last_sync_summary(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM sync_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.conn.close()