"""SQLite persistence for per-vessel AIS snapshots and review batches.

``vessel_state``: one row per ``(source_id, mmsi, kind)`` where ``kind`` is
``'position'`` (message types 1/2/3) or ``'static'`` (message type 5). A
stored row is only replaced by a *strictly newer* ``observed_at``:

* older batches never roll back newer state;
* an equal ``observed_at`` keeps the row that was saved first;
* inside one batch (all messages share the batch ``observed_at``) the
  caller applies decoded messages in sentence-index order, so the first
  message of a class wins.

Whole decoded messages are stored as-is — a position report whose
coordinates are "not available" (null) is stored with null coordinates;
no fields are borrowed from older messages.

``vessel_track``: append-only time series of every stored position report
(message types 1/2/3), one row per ``(source_id, mmsi, observed_at)`` —
late (older) observations are kept in the track even though they never
roll back the ``vessel_state`` snapshot, and a repeated submission at an
already-stored ``observed_at`` does not add another point (``INSERT OR
IGNORE`` keeps the first saved message). Each point records the original
sentence index for provenance. When an older database without this table
is opened, each vessel's existing position snapshot is copied in as a
single starting point (``sentence_index = -1``); no earlier track is
inferred.

A track point found to be a misreport is marked invalid in place
(``valid = 0`` plus the ``invalid_reason``/``invalidated_at`` record);
the row is never deleted. Invalid points are hidden from ``GET /tracks``,
and the marking, the visibility change and the snapshot fall-back commit
in one transaction:

* when the ``vessel_state`` position snapshot is exactly the invalidated
  point, it falls back to the vessel's latest still-valid track point,
  or the position part is removed when no valid point remains (the
  static part is never touched);
* invalidating an older point never rolls back a newer valid snapshot;
* a later ``/ingest`` or review resolution carrying a position report at
  an invalidated ``(source_id, mmsi, observed_at)`` can neither
  re-activate the point nor write it back into the snapshot — the write
  is skipped, while every other observation follows the usual rules.

Databases written before the invalidation columns existed are migrated
in place (``ALTER TABLE``) when the store is opened.

``position_conflict``: one row per *source pair* reporting the same MMSI
at the same UTC ``observed_at`` at positions strictly more than one
nautical mile (1852 m) apart. Only track points that are actually stored,
still valid and carry complete coordinates participate. The list is a
derived view of ``vessel_track``: every write path and every
invalidation refreshes the affected ``(mmsi, observed_at)`` groups in
the same transaction, and opening any database rebuilds the whole list,
so older databases surface their conflicts after an upgrade and results
stay identical across restarts. Invalidating one point removes exactly
the pairs involving it; pairs of the other sources are kept.

``jump_segment``: one row per pair of *consecutive* valid track points of
one ``(source_id, mmsi)`` whose implied average speed — great-circle
distance over the ``observed_at`` interval — is strictly greater than 50
knots. Both ends must carry complete coordinates and a positive time
interval; a point with missing coordinates stays in the track but breaks
adjacency, so no segment is ever stitched across it. The list is a
derived view of ``vessel_track`` maintained per vessel inside the same
transactions as track writes, review resolutions and invalidations (a
failure rolls all back), and rebuilt wholesale when a database is opened,
so tracks stored by an older database surface their segments after the
upgrade. A late observation or a late review resolution splits the
segment it lands in; invalidating a middle point joins its two
neighbours; a same-instant retransmission and an unchosen ambiguous
candidate change nothing.

``review_batch``: one row per manual-review batch created from an ambiguous
batch of sentences. The original sentences and the full decode outcome
(including every candidate of every ambiguous pool) are stored as JSON, so
pending batches, their candidates and the resolution (chosen candidates plus
the snapshots actually stored) survive restarts. Resolution applies the
snapshot and track writes and flips the status in a single transaction.

A read-only impact preview (:meth:`SnapshotStore.preview_ingest`) runs the
exact write pipeline of a normal ingest inside one locked transaction, reads
the before/after views from that same transaction, and always rolls back —
so a preview reports only real changes and leaves no write behind whether it
succeeds or fails. The affected ``(mmsi, observed_at)`` groups of a batch are
staged in a per-connection temporary table that the before/after reads match
against, so a batch touching arbitrarily many groups (a full 2000-sentence
submission can carry well over a thousand) never hits SQLite's
expression-depth limit.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .models import (
    JumpSegment,
    JumpSegmentEndpoint,
    NewTrackPoint,
    PositionConflict,
    PositionConflictPoint,
    PositionReport,
    SnapshotChange,
    StaticAndVoyageData,
    TimedPositionReport,
    TimedStaticAndVoyageData,
    TrackPoint,
    VesselSnapshot,
)

KIND_POSITION = "position"
KIND_STATIC = "static"

# Two simultaneous reports of one MMSI are a conflict only when strictly
# farther apart than one nautical mile (the haversine comparison is
# ``>``; exactly 1852 m is not a conflict).
CONFLICT_DISTANCE_METRES = 1852.0
EARTH_RADIUS_METRES = 6_371_000.0

# Two consecutive valid track points of one vessel form a suspected jump
# segment only when the average speed implied by the great-circle distance
# is strictly greater than 50 knots (the comparison is ``>``; exactly
# 50 knots is not listed).
JUMP_SPEED_KNOTS = 50.0
METRES_PER_NAUTICAL_MILE = 1852.0

# sentence_index recorded for track points backfilled from pre-existing
# vessel_state rows of an older database (no earlier track is inferred).
LEGACY_TRACK_INDEX = -1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vessel_state (
    source_id   TEXT NOT NULL,
    mmsi        INTEGER NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('position', 'static')),
    observed_at TEXT NOT NULL,
    data        TEXT NOT NULL,
    PRIMARY KEY (source_id, mmsi, kind)
);

CREATE TABLE IF NOT EXISTS vessel_track (
    source_id      TEXT NOT NULL,
    mmsi           INTEGER NOT NULL,
    observed_at    TEXT NOT NULL,
    sentence_index INTEGER NOT NULL,
    data           TEXT NOT NULL,
    valid          INTEGER NOT NULL DEFAULT 1,
    invalid_reason TEXT,
    invalidated_at TEXT,
    PRIMARY KEY (source_id, mmsi, observed_at)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS review_batch (
    batch_id     TEXT PRIMARY KEY,
    status       TEXT NOT NULL CHECK (status IN ('pending', 'resolved')),
    observed_at  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    resolved_at  TEXT,
    request      TEXT NOT NULL,
    results      TEXT NOT NULL,
    resolution   TEXT
);

CREATE TABLE IF NOT EXISTS position_conflict (
    mmsi         INTEGER NOT NULL,
    observed_at  TEXT NOT NULL,
    source_id_a  TEXT NOT NULL,
    source_id_b  TEXT NOT NULL,
    longitude_a  REAL NOT NULL,
    latitude_a   REAL NOT NULL,
    longitude_b  REAL NOT NULL,
    latitude_b   REAL NOT NULL,
    distance_m   REAL NOT NULL,
    PRIMARY KEY (mmsi, observed_at, source_id_a, source_id_b)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS jump_segment (
    source_id         TEXT NOT NULL,
    mmsi              INTEGER NOT NULL,
    start_observed_at TEXT NOT NULL,
    end_observed_at   TEXT NOT NULL,
    start_longitude   REAL NOT NULL,
    start_latitude    REAL NOT NULL,
    end_longitude     REAL NOT NULL,
    end_latitude      REAL NOT NULL,
    distance_m        REAL NOT NULL,
    speed_knots       REAL NOT NULL,
    PRIMARY KEY (source_id, mmsi, start_observed_at, end_observed_at)
) WITHOUT ROWID;
"""

# Per-connection staging table (temp schema) holding the (mmsi, observed_at)
# groups a preview read targets. Matching the before/after reads against it
# keeps arbitrarily large group sets queryable without stitching one OR
# clause per group into a single expression — SQLite caps expression depth
# (1000 by default), which a batch of ~1000 distinct groups would exceed.
_GROUP_STAGING_SCHEMA = """
CREATE TEMP TABLE IF NOT EXISTS affected_group (
    mmsi        INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (mmsi, observed_at)
)
"""

# Existing position snapshots of an older database become one starting
# track point per vessel; INSERT OR IGNORE makes this idempotent on
# databases that already carry track points.
_BACKFILL_TRACK_SQL = """
INSERT OR IGNORE INTO vessel_track
    (source_id, mmsi, observed_at, sentence_index, data)
SELECT source_id, mmsi, observed_at, :legacy_index, data
FROM vessel_state
WHERE kind = 'position'
"""

# Databases written before track-point invalidation existed are migrated
# in place: the invalidation columns are added when missing (existing
# points stay valid).
_MIGRATE_TRACK_COLUMNS = (
    ("valid", "ALTER TABLE vessel_track ADD COLUMN valid INTEGER NOT NULL DEFAULT 1"),
    ("invalid_reason", "ALTER TABLE vessel_track ADD COLUMN invalid_reason TEXT"),
    ("invalidated_at", "ALTER TABLE vessel_track ADD COLUMN invalidated_at TEXT"),
)

_INSERT_TRACK_SQL = """
INSERT OR IGNORE INTO vessel_track
    (source_id, mmsi, observed_at, sentence_index, data)
VALUES (?, ?, ?, ?, ?)
"""

_UPSERT_SQL = """
INSERT INTO vessel_state (source_id, mmsi, kind, observed_at, data)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (source_id, mmsi, kind) DO UPDATE SET
    observed_at = excluded.observed_at,
    data = excluded.data
WHERE excluded.observed_at > vessel_state.observed_at
"""

# The conflict list is recomputed per (mmsi, observed_at) group: the
# stored pairs are deleted first, then every pair of valid points from
# distinct sources with complete coordinates strictly farther apart than
# one nautical mile is inserted. The ordering constraint makes each source
# pair canonical (source_id_a < source_id_b).
_DELETE_CONFLICTS_FOR_GROUP_SQL = """
DELETE FROM position_conflict
WHERE mmsi = ? AND observed_at = ?
"""

_VALID_POSITIONS_FOR_GROUP_SQL = """
SELECT source_id, data
FROM vessel_track
WHERE mmsi = ? AND observed_at = ? AND valid = 1
"""

# Valid track points of one vessel in track order (the same order
# ``GET /tracks`` returns them); consecutive pairs are the jump-segment
# candidates.
_VALID_POINTS_FOR_VESSEL_SQL = """
SELECT observed_at, data
FROM vessel_track
WHERE source_id = ? AND mmsi = ? AND valid = 1
ORDER BY observed_at, sentence_index
"""

_DELETE_JUMP_SEGMENTS_FOR_VESSEL_SQL = """
DELETE FROM jump_segment
WHERE source_id = ? AND mmsi = ?
"""

_INSERT_JUMP_SEGMENT_SQL = """
INSERT INTO jump_segment
    (source_id, mmsi, start_observed_at, end_observed_at,
     start_longitude, start_latitude, end_longitude, end_latitude,
     distance_m, speed_knots)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def normalize_observed_at(value: datetime) -> str:
    """Fixed-width UTC ISO 8601 string; lexicographic order == time order."""
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_observed_at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def haversine_metres(
    lon_a: float, lat_a: float, lon_b: float, lat_b: float
) -> float:
    """Great-circle distance between two WGS-84 coordinates, in metres."""
    lon1, lat1, lon2, lat2 = map(
        math.radians, (lon_a, lat_a, lon_b, lat_b)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METRES * math.asin(math.sqrt(a))


class SnapshotStore:
    """Thread-safe SQLite-backed store of the latest message per vessel."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            # Per-connection staging table for the preview's group reads
            # (temp schema: lives and dies with this connection).
            self._conn.execute(_GROUP_STAGING_SCHEMA)
            # One-time migration for older databases: existing position
            # snapshots seed each vessel's track with a single starting point.
            self._conn.execute(
                _BACKFILL_TRACK_SQL, {"legacy_index": LEGACY_TRACK_INDEX}
            )
            # One-time migration for older databases: add the invalidation
            # columns to vessel_track when they are missing.
            columns = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(vessel_track)")
            }
            for column, ddl in _MIGRATE_TRACK_COLUMNS:
                if column not in columns:
                    self._conn.execute(ddl)
            # The conflict list is a derived view of vessel_track: rebuild it
            # wholesale when opening an older database (missing conflicts for
            # simultaneous reports are backfilled even when the track rows
            # already existed), and idempotently in every other case.
            self._rebuild_all_conflicts()
            # The jump-segment list is a derived view of vessel_track too:
            # rebuilt wholesale on open so tracks stored by an older
            # database surface their suspected jump segments, idempotently
            # on every later open.
            self._rebuild_all_jump_segments()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes -------------------------------------------------------------

    def _point_invalidated(
        self, source_id: str, mmsi: int, observed_at: str
    ) -> bool:
        """True when the track point for this exact key was marked invalid.

        Caller must hold the lock.
        """
        row = self._conn.execute(
            """
            SELECT valid FROM vessel_track
            WHERE source_id = ? AND mmsi = ? AND observed_at = ?
            """,
            (source_id, mmsi, observed_at),
        ).fetchone()
        return row is not None and not row["valid"]

    def _write_message(
        self,
        source_id: str,
        mmsi: int,
        kind: str,
        observed_at: str,
        data: dict,
        sentence_index: int = 0,
        affected: Optional[set[tuple[int, str]]] = None,
    ) -> bool:
        """Upsert the snapshot row and record a position track point.

        Every position report (kind == 'position') is appended to
        ``vessel_track`` regardless of whether it also replaces the
        snapshot: a late older observation still enters the track, while
        ``INSERT OR IGNORE`` keeps the first saved message for an already
        known ``(source_id, mmsi, observed_at)``. A position report whose
        ``(source_id, mmsi, observed_at)`` was marked invalid is skipped
        entirely — an invalidated misreport is never re-activated or
        written back into the snapshot. Caller must hold the lock and the
        transaction.

        Every attempted position write (including a skipped retransmission
        or an invalidated-key write) registers its ``(mmsi, observed_at)``
        group in *affected*, so the caller can refresh exactly the
        position-conflict pairs the batch could have changed.
        """
        if kind == KIND_POSITION:
            if affected is not None:
                affected.add((mmsi, observed_at))
            if self._point_invalidated(source_id, mmsi, observed_at):
                return False
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
        cursor = self._conn.execute(
            _UPSERT_SQL, (source_id, mmsi, kind, observed_at, payload)
        )
        if kind == KIND_POSITION:
            self._conn.execute(
                _INSERT_TRACK_SQL,
                (source_id, mmsi, observed_at, sentence_index, payload),
            )
        return cursor.rowcount > 0

    # -- position conflicts -----------------------------------------------------

    @staticmethod
    def _coordinates(data: str) -> Optional[tuple[float, float]]:
        """``(longitude, latitude)`` from a stored track payload, else None."""
        message = json.loads(data)
        longitude = message.get("longitude")
        latitude = message.get("latitude")
        if longitude is None or latitude is None:
            return None
        return longitude, latitude

    def _refresh_conflicts(
        self, groups: set[tuple[int, str]]
    ) -> None:
        """Recompute every source-pair conflict of the given time groups.

        For each ``(mmsi, observed_at)`` the stored pairs are deleted and
        re-created from the valid track points of *distinct* sources that
        carry complete coordinates and are strictly farther apart than
        one nautical mile. The recomputation is a derived, idempotent view
        of ``vessel_track``; the caller runs it inside the write
        transaction so conflict rows commit together with track writes,
        review status or an invalidation (a failure rolls all back).
        """
        for mmsi, observed_at in groups:
            self._conn.execute(
                _DELETE_CONFLICTS_FOR_GROUP_SQL, (mmsi, observed_at)
            )
            points: list[tuple[str, float, float]] = []
            for row in self._conn.execute(
                _VALID_POSITIONS_FOR_GROUP_SQL, (mmsi, observed_at)
            ):
                coordinates = self._coordinates(row["data"])
                if coordinates is not None:
                    points.append((row["source_id"], *coordinates))
            points.sort(key=lambda item: item[0])
            for i in range(len(points)):
                source_a, lon_a, lat_a = points[i]
                for j in range(i + 1, len(points)):
                    source_b, lon_b, lat_b = points[j]
                    distance = haversine_metres(lon_a, lat_a, lon_b, lat_b)
                    if distance > CONFLICT_DISTANCE_METRES:
                        self._conn.execute(
                            """
                            INSERT INTO position_conflict
                                (mmsi, observed_at, source_id_a, source_id_b,
                                 longitude_a, latitude_a, longitude_b, latitude_b,
                                 distance_m)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                mmsi, observed_at, source_a, source_b,
                                lon_a, lat_a, lon_b, lat_b, distance,
                            ),
                        )

    def _rebuild_all_conflicts(self) -> None:
        """Recompute the full conflict list from every stored track point."""
        groups = {
            (row["mmsi"], row["observed_at"])
            for row in self._conn.execute(
                "SELECT DISTINCT mmsi, observed_at FROM vessel_track"
            )
        }
        self._refresh_conflicts(groups)

    @staticmethod
    def _conflict_from_row(row: sqlite3.Row) -> PositionConflict:
        return PositionConflict(
            mmsi=row["mmsi"],
            observed_at=_parse_observed_at(row["observed_at"]),
            source_id_a=row["source_id_a"],
            point_a=PositionConflictPoint(
                longitude=row["longitude_a"], latitude=row["latitude_a"]
            ),
            source_id_b=row["source_id_b"],
            point_b=PositionConflictPoint(
                longitude=row["longitude_b"], latitude=row["latitude_b"]
            ),
            distance_m=round(row["distance_m"], 3),
        )

    # -- jump segments --------------------------------------------------------

    def _refresh_jump_segments(self, vessels: set[tuple[str, int]]) -> None:
        """Recompute the suspected jump segments of the given vessels.

        For each ``(source_id, mmsi)`` the stored segments are deleted and
        re-created from the *consecutive* valid track points: a pair is
        listed only when both ends carry complete coordinates, the
        ``observed_at`` interval is positive and the implied average speed
        (great-circle distance over time) is strictly greater than 50
        knots. A point with missing coordinates stays in the track but
        breaks adjacency — the iteration never stitches a segment across
        it. The recomputation is a derived, idempotent view of
        ``vessel_track``; the caller runs it inside the write transaction
        so segment rows commit together with track writes, a review
        resolution or an invalidation (a failure rolls all back).
        """
        for source_id, mmsi in vessels:
            self._conn.execute(
                _DELETE_JUMP_SEGMENTS_FOR_VESSEL_SQL, (source_id, mmsi)
            )
            rows = list(
                self._conn.execute(
                    _VALID_POINTS_FOR_VESSEL_SQL, (source_id, mmsi)
                )
            )
            # (observed_at, longitude, latitude) of the previous valid
            # point with complete coordinates; None after a point with
            # missing coordinates, so no segment spans such a gap.
            previous: Optional[tuple[str, float, float]] = None
            for row in rows:
                coordinates = self._coordinates(row["data"])
                current = (
                    (row["observed_at"], *coordinates)
                    if coordinates is not None
                    else None
                )
                if previous is not None and current is not None:
                    start_at, lon_a, lat_a = previous
                    end_at, lon_b, lat_b = current
                    seconds = (
                        _parse_observed_at(end_at) - _parse_observed_at(start_at)
                    ).total_seconds()
                    if seconds > 0:
                        distance = haversine_metres(lon_a, lat_a, lon_b, lat_b)
                        speed = (distance / METRES_PER_NAUTICAL_MILE) / (
                            seconds / 3600.0
                        )
                        if speed > JUMP_SPEED_KNOTS:
                            self._conn.execute(
                                _INSERT_JUMP_SEGMENT_SQL,
                                (
                                    source_id, mmsi, start_at, end_at,
                                    lon_a, lat_a, lon_b, lat_b,
                                    distance, speed,
                                ),
                            )
                previous = current

    def _rebuild_all_jump_segments(self) -> None:
        """Recompute the full jump-segment list from every stored track."""
        vessels = {
            (row["source_id"], row["mmsi"])
            for row in self._conn.execute(
                "SELECT DISTINCT source_id, mmsi FROM vessel_track"
            )
        }
        self._refresh_jump_segments(vessels)

    @staticmethod
    def _jump_segment_from_row(row: sqlite3.Row) -> JumpSegment:
        return JumpSegment(
            source_id=row["source_id"],
            mmsi=row["mmsi"],
            start=JumpSegmentEndpoint(
                observed_at=_parse_observed_at(row["start_observed_at"]),
                longitude=row["start_longitude"],
                latitude=row["start_latitude"],
            ),
            end=JumpSegmentEndpoint(
                observed_at=_parse_observed_at(row["end_observed_at"]),
                longitude=row["end_longitude"],
                latitude=row["end_latitude"],
            ),
            distance_m=round(row["distance_m"], 3),
            average_speed_knots=round(row["speed_knots"], 3),
        )

    def list_jump_segments(
        self,
        source_id: Optional[str] = None,
        mmsi: Optional[int] = None,
    ) -> list[JumpSegment]:
        """Stored suspected jump segments, ordered by end time, MMSI, source.

        Only consecutive valid track points with complete coordinates on
        both ends, a positive time interval and an implied average speed
        strictly greater than 50 knots are listed. Both filters are
        optional; omitted filters list every stored segment.
        """
        sql = "SELECT * FROM jump_segment"
        clauses: list[str] = []
        params: list = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if mmsi is not None:
            clauses.append("mmsi = ?")
            params.append(mmsi)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY end_observed_at, mmsi, source_id, start_observed_at"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._jump_segment_from_row(row) for row in rows]

    @dataclass
    class IngestPreviewDiff:
        """Actual changes one :meth:`preview_ingest` dry run would produce."""

        snapshot_changes: list[SnapshotChange]
        new_track_points: list[NewTrackPoint]
        new_position_conflicts: list[PositionConflict]

    def _stage_groups(self, groups: set[tuple[int, str]]) -> None:
        """Load *groups* into the connection's group-staging table.

        Caller must hold the lock. The table is re-filled before every
        read, and a preview's rollback also reverts its contents, so no
        stale rows can leak from one read into the next.
        """
        self._conn.execute("DELETE FROM affected_group")
        self._conn.executemany(
            "INSERT INTO affected_group (mmsi, observed_at) VALUES (?, ?)",
            groups,
        )

    def _conflicts_for_groups(
        self, groups: set[tuple[int, str]]
    ) -> list[PositionConflict]:
        """Current conflict rows of *groups*; caller must hold the lock."""
        if not groups:
            return []
        self._stage_groups(groups)
        rows = self._conn.execute(
            """
            SELECT * FROM position_conflict
            WHERE (mmsi, observed_at) IN
                  (SELECT mmsi, observed_at FROM affected_group)
            """
        ).fetchall()
        return [self._conflict_from_row(row) for row in rows]

    def _track_points_for_groups(
        self, groups: set[tuple[int, str]]
    ) -> list[sqlite3.Row]:
        """Valid track rows of the given ``(mmsi, observed_at)`` groups.

        Ordered like ``GET /tracks`` output; caller must hold the lock.
        """
        if not groups:
            return []
        self._stage_groups(groups)
        return list(
            self._conn.execute(
                """
                SELECT source_id, mmsi, observed_at, sentence_index, data
                FROM vessel_track
                WHERE valid = 1 AND (mmsi, observed_at) IN
                      (SELECT mmsi, observed_at FROM affected_group)
                ORDER BY mmsi, observed_at, source_id, sentence_index
                """
            ).fetchall()
        )

    @staticmethod
    def _conflict_key(conflict: PositionConflict) -> tuple[int, str, str, str]:
        observed_at = normalize_observed_at(conflict.observed_at)
        return (
            conflict.mmsi, observed_at, conflict.source_id_a, conflict.source_id_b,
        )

    def preview_ingest(
        self,
        writes: list[tuple[str, int, str, str, dict, int]],
    ) -> "SnapshotStore.IngestPreviewDiff":
        """Dry-run one ``/ingest`` batch without persisting anything.

        The exact same write pipeline as :meth:`apply_ingest` (strictly-newer
        snapshot upserts, ``INSERT OR IGNORE`` track points, invalidated-key
        skips and per-group conflict recomputation) is executed inside one
        locked transaction; before/after views are read from that same
        transaction and the transaction is then always rolled back, so a
        successful or a failing preview leaves no write behind.

        Only actual changes are returned: per-``(source_id, MMSI)`` merged
        snapshots that differ, valid track points that would be newly
        inserted (a late older observation qualifies), and same-instant
        source-pair conflicts that would be newly created.
        """
        with self._lock:
            involved = sorted({(w[0], w[1]) for w in writes})
            groups: set[tuple[int, str]] = {
                (w[1], w[3]) for w in writes if w[2] == KIND_POSITION
            }

            try:
                self._conn.execute("BEGIN")

                # Before-image and after-image are read inside this one
                # transaction, hence from the same SQLite snapshot.
                before_rows: dict[tuple[str, int], list[sqlite3.Row]] = {
                    (source_id, mmsi): self._fetch_rows(source_id, mmsi)
                    for source_id, mmsi in involved
                }
                # Only the (mmsi, observed_at) groups this batch writes can
                # change; capture their valid track points on both sides.
                before_point_rows = self._track_points_for_groups(groups)
                before_conflicts = {
                    self._conflict_key(conflict): conflict
                    for conflict in self._conflicts_for_groups(groups)
                }

                affected: set[tuple[int, str]] = set()
                for source_id, mmsi, kind, observed_at, data, sentence_index in writes:
                    self._write_message(
                        source_id, mmsi, kind, observed_at, data,
                        sentence_index, affected=affected,
                    )
                self._refresh_conflicts(affected)
                # The same jump-segment refresh a real ingest would run; the
                # rollback below reverts it together with every other write.
                self._refresh_jump_segments(
                    {(w[0], w[1]) for w in writes if w[2] == KIND_POSITION}
                )

                after_rows: dict[tuple[str, int], list[sqlite3.Row]] = {
                    (source_id, mmsi): self._fetch_rows(source_id, mmsi)
                    for source_id, mmsi in involved
                }
                # Compare the post-image valid points with the before points
                # captured at the start of this same transaction.
                after_point_rows = self._track_points_for_groups(affected)
                after_conflicts = {
                    self._conflict_key(conflict): conflict
                    for conflict in self._conflicts_for_groups(affected)
                }
            finally:
                # A preview never commits: roll back on success and on error.
                self._conn.rollback()

            # Snapshot before/after diffs (whole merged snapshots).
            snapshot_changes: list[SnapshotChange] = []
            for source_id, mmsi in involved:
                before = (
                    self._to_snapshot(source_id, mmsi, before_rows[(source_id, mmsi)])
                    if before_rows[(source_id, mmsi)]
                    else None
                )
                after = (
                    self._to_snapshot(source_id, mmsi, after_rows[(source_id, mmsi)])
                    if after_rows[(source_id, mmsi)]
                    else None
                )
                before_json = before.model_dump(mode="json") if before else None
                after_json = after.model_dump(mode="json") if after else None
                if after_json != before_json:
                    snapshot_changes.append(
                        SnapshotChange(
                            source_id=source_id, mmsi=mmsi,
                            before=before, after=after,
                        )
                    )

            # Newly inserted valid track points (skip same-key rewrites such
            # as same-instant retransmissions and invalidated keys).
            before_track_keys = {
                (row["source_id"], row["mmsi"], row["observed_at"])
                for row in before_point_rows
            }
            new_track_points: list[NewTrackPoint] = []
            for row in after_point_rows:
                key = (row["source_id"], row["mmsi"], row["observed_at"])
                if key in before_track_keys:
                    continue
                new_track_points.append(
                    NewTrackPoint(
                        source_id=row["source_id"],
                        mmsi=row["mmsi"],
                        point=TrackPoint(
                            observed_at=_parse_observed_at(row["observed_at"]),
                            data=PositionReport(**json.loads(row["data"])),
                        ),
                    )
                )

            # Newly created same-instant source-pair conflicts, in the same
            # order as GET /position-conflicts (time, MMSI, source pair).
            new_position_conflicts = sorted(
                (
                    conflict
                    for key, conflict in after_conflicts.items()
                    if key not in before_conflicts
                ),
                key=lambda conflict: (
                    normalize_observed_at(conflict.observed_at),
                    conflict.mmsi,
                    conflict.source_id_a,
                    conflict.source_id_b,
                ),
            )

            return self.IngestPreviewDiff(
                snapshot_changes=snapshot_changes,
                new_track_points=new_track_points,
                new_position_conflicts=new_position_conflicts,
            )

    def upsert_if_newer(
        self,
        source_id: str,
        mmsi: int,
        kind: str,
        observed_at: str,
        data: dict,
    ) -> bool:
        """Store *data* unless an equal-or-newer snapshot row already exists.

        Position reports additionally append a track point (late reports
        still enter the track; a repeated submission at the same
        ``observed_at`` does not duplicate it). *observed_at* must be
        normalised with :func:`normalize_observed_at`. Returns True when
        the snapshot row was (re)written. The position-conflict groups and
        the jump segments the write can change are refreshed in the same
        transaction.
        """
        with self._lock, self._conn:
            affected: set[tuple[int, str]] = set()
            changed = self._write_message(
                source_id, mmsi, kind, observed_at, data, affected=affected
            )
            self._refresh_conflicts(affected)
            if kind == KIND_POSITION:
                self._refresh_jump_segments({(source_id, mmsi)})
            return changed

    def apply_ingest(
        self,
        writes: list[tuple[str, int, str, str, dict, int]],
    ) -> list[tuple[str, int]]:
        """Apply one normal ``/ingest`` batch in a single transaction.

        *writes* are
        ``(source_id, mmsi, kind, observed_at, data, sentence_index)``
        tuples, already ordered by original sentence index. Snapshot
        upserts, position track inserts, position-conflict refreshes and
        jump-segment refreshes commit or roll back together. Returns the
        involved vessels' ``(source_id, mmsi)`` pairs.
        """
        with self._lock, self._conn:
            affected: set[tuple[int, str]] = set()
            for source_id, mmsi, kind, observed_at, data, sentence_index in writes:
                self._write_message(
                    source_id, mmsi, kind, observed_at, data,
                    sentence_index, affected=affected,
                )
            self._refresh_conflicts(affected)
            self._refresh_jump_segments(
                {(w[0], w[1]) for w in writes if w[2] == KIND_POSITION}
            )
            return sorted({(w[0], w[1]) for w in writes})

    # -- reads ----------------------------------------------------------------

    def _fetch_rows(
        self,
        source_id: Optional[str] = None,
        mmsi: Optional[int] = None,
    ) -> list[sqlite3.Row]:
        """SELECT matching ``vessel_state`` rows; caller must hold the lock."""
        sql = "SELECT source_id, mmsi, kind, observed_at, data FROM vessel_state"
        clauses: list[str] = []
        params: list = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if mmsi is not None:
            clauses.append("mmsi = ?")
            params.append(mmsi)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY source_id, mmsi, kind"
        return list(self._conn.execute(sql, params).fetchall())

    def _rows(
        self,
        source_id: Optional[str] = None,
        mmsi: Optional[int] = None,
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self._fetch_rows(source_id, mmsi)

    @staticmethod
    def _to_snapshot(
        source_id: str, mmsi: int, rows: list[sqlite3.Row]
    ) -> VesselSnapshot:
        snapshot = VesselSnapshot(source_id=source_id, mmsi=mmsi)
        for row in rows:
            observed_at = _parse_observed_at(row["observed_at"])
            data = json.loads(row["data"])
            if row["kind"] == KIND_POSITION:
                snapshot.position = TimedPositionReport(
                    observed_at=observed_at, data=PositionReport(**data)
                )
            else:
                snapshot.static = TimedStaticAndVoyageData(
                    observed_at=observed_at, data=StaticAndVoyageData(**data)
                )
        return snapshot

    def get_snapshot(self, source_id: str, mmsi: int) -> Optional[VesselSnapshot]:
        rows = self._rows(source_id=source_id, mmsi=mmsi)
        if not rows:
            return None
        return self._to_snapshot(source_id, mmsi, rows)

    def list_snapshots(
        self,
        source_id: Optional[str] = None,
        mmsi: Optional[int] = None,
    ) -> list[VesselSnapshot]:
        grouped: dict[tuple[str, int], list[sqlite3.Row]] = {}
        for row in self._rows(source_id=source_id, mmsi=mmsi):
            grouped.setdefault((row["source_id"], row["mmsi"]), []).append(row)
        return [
            self._to_snapshot(src, mmsi, rows)
            for (src, mmsi), rows in sorted(grouped.items())
        ]

    # -- track ----------------------------------------------------------------

    def vessel_exists(self, source_id: str, mmsi: int) -> bool:
        """True when the vessel has any stored snapshot or track point."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM vessel_state WHERE source_id = ? AND mmsi = ? LIMIT 1",
                (source_id, mmsi),
            ).fetchone()
            if row is not None:
                return True
            row = self._conn.execute(
                "SELECT 1 FROM vessel_track WHERE source_id = ? AND mmsi = ? LIMIT 1",
                (source_id, mmsi),
            ).fetchone()
            return row is not None

    def get_track(self, source_id: str, mmsi: int) -> list[TrackPoint]:
        """All valid stored position reports for one vessel, oldest first.

        Ordered by ``observed_at`` (then original sentence index); points
        are stored as decoded, including null coordinates. Points marked
        invalid (misreports) are not returned.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT observed_at, data FROM vessel_track
                WHERE source_id = ? AND mmsi = ? AND valid = 1
                ORDER BY observed_at, sentence_index
                """,
                (source_id, mmsi),
            ).fetchall()
        return [
            TrackPoint(
                observed_at=_parse_observed_at(row["observed_at"]),
                data=PositionReport(**json.loads(row["data"])),
            )
            for row in rows
        ]

    # -- position conflicts -----------------------------------------------------

    def list_position_conflicts(
        self,
        source_id: Optional[str] = None,
        mmsi: Optional[int] = None,
    ) -> list[PositionConflict]:
        """Stored same-instant, far-apart source pairs, oldest conflict first.

        Only points actually kept in ``vessel_track`` (not invalidated) with
        complete coordinates on both sides participate, and pairs are
        listed only when the great-circle distance is strictly greater
        than one nautical mile. Ordered by ``observed_at``, then MMSI,
        then the two source ids. An optional *source_id* filter matches a
        conflict on either side; an optional *mmsi* filter narrows the
        vessel.
        """
        sql = "SELECT * FROM position_conflict"
        clauses: list[str] = []
        params: list = []
        if source_id is not None:
            clauses.append("(source_id_a = ? OR source_id_b = ?)")
            params.extend((source_id, source_id))
        if mmsi is not None:
            clauses.append("mmsi = ?")
            params.append(mmsi)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += (
            " ORDER BY observed_at, mmsi, source_id_a, source_id_b"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._conflict_from_row(row) for row in rows]

    # -- track point invalidation ----------------------------------------------

    def _fallback_position_snapshot(
        self, source_id: str, mmsi: int, observed_at: str
    ) -> None:
        """Fall the position snapshot back to the latest valid track point.

        Only acts when the stored position snapshot is exactly the point
        being invalidated — invalidating an older point never rolls back a
        newer valid snapshot. When no valid point remains, the position
        part is removed (the static part is untouched). Caller must hold
        the lock and the transaction.
        """
        row = self._conn.execute(
            """
            SELECT observed_at FROM vessel_state
            WHERE source_id = ? AND mmsi = ? AND kind = ?
            """,
            (source_id, mmsi, KIND_POSITION),
        ).fetchone()
        if row is None or row["observed_at"] != observed_at:
            return
        fallback = self._conn.execute(
            """
            SELECT observed_at, data FROM vessel_track
            WHERE source_id = ? AND mmsi = ? AND valid = 1
            ORDER BY observed_at DESC
            LIMIT 1
            """,
            (source_id, mmsi),
        ).fetchone()
        if fallback is None:
            self._conn.execute(
                """
                DELETE FROM vessel_state
                WHERE source_id = ? AND mmsi = ? AND kind = ?
                """,
                (source_id, mmsi, KIND_POSITION),
            )
        else:
            self._conn.execute(
                """
                UPDATE vessel_state
                SET observed_at = ?, data = ?
                WHERE source_id = ? AND mmsi = ? AND kind = ?
                """,
                (
                    fallback["observed_at"],
                    fallback["data"],
                    source_id,
                    mmsi,
                    KIND_POSITION,
                ),
            )

    def invalidate_point(
        self,
        source_id: str,
        mmsi: int,
        observed_at: str,
        reason: str,
        invalidated_at: str,
    ) -> Optional[tuple[TrackPoint, str, str]]:
        """Mark one stored track point as a misreport, atomically.

        The point stays stored but becomes invisible to ``GET /tracks``;
        when the position snapshot is exactly this point it falls back to
        the latest valid point (or is removed). The marking, the
        visibility change and the snapshot fall-back commit in one
        transaction. Returns ``(original_point, reason, invalidated_at)``
        of the *first* invalidation record — a repeated marking changes
        nothing and returns the first record — or ``None`` when no track
        point exists for the key.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                """
                SELECT data, valid, invalid_reason, invalidated_at
                FROM vessel_track
                WHERE source_id = ? AND mmsi = ? AND observed_at = ?
                """,
                (source_id, mmsi, observed_at),
            ).fetchone()
            if row is None:
                return None
            point = TrackPoint(
                observed_at=_parse_observed_at(observed_at),
                data=PositionReport(**json.loads(row["data"])),
            )
            if not row["valid"]:
                # already invalid: the first record wins, nothing changes
                return point, row["invalid_reason"], row["invalidated_at"]
            self._conn.execute(
                """
                UPDATE vessel_track
                SET valid = 0, invalid_reason = ?, invalidated_at = ?
                WHERE source_id = ? AND mmsi = ? AND observed_at = ?
                """,
                (reason, invalidated_at, source_id, mmsi, observed_at),
            )
            self._fallback_position_snapshot(source_id, mmsi, observed_at)
            # Pairs involving the now-invalid point are removed; pairs of
            # the other sources at this time are recomputed and kept.
            self._refresh_conflicts({(mmsi, observed_at)})
            # Removing the point from the valid track can join its two
            # neighbours into one suspected jump segment.
            self._refresh_jump_segments({(source_id, mmsi)})
            return point, reason, invalidated_at

    # -- review batches -------------------------------------------------------

    def create_review(
        self,
        batch_id: str,
        observed_at: str,
        created_at: str,
        request_json: str,
        results_json: str,
    ) -> None:
        """Store a new pending review batch (snapshots are not touched)."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO review_batch
                    (batch_id, status, observed_at, created_at, request, results)
                VALUES (?, 'pending', ?, ?, ?, ?)
                """,
                (batch_id, observed_at, created_at, request_json, results_json),
            )

    def get_review(self, batch_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM review_batch WHERE batch_id = ?", (batch_id,)
            ).fetchone()

    def list_reviews(self, status: Optional[str] = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM review_batch"
        params: list = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at, batch_id"
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def resolve_review(
        self,
        batch_id: str,
        resolved_at: str,
        selections: list[list[int]],
        writes: list[tuple[str, int, str, str, dict, int]],
    ) -> Optional[list[VesselSnapshot]]:
        """Apply snapshot/track *writes* and mark the batch resolved, atomically.

        *writes* are
        ``(source_id, mmsi, kind, observed_at, data, sentence_index)``
        tuples in original-sentence-index order; each is applied with the
        usual strictly-newer snapshot rule (newer snapshots are never
        rolled back), while every position report also enters the track.
        The status flip, every snapshot write and every track insert
        commit or roll back together. Returns the involved vessels'
        snapshots as actually stored, or ``None`` when the batch is
        unknown or already resolved (nothing is written then).
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT status FROM review_batch WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None or row["status"] != "pending":
                return None
            affected: set[tuple[int, str]] = set()
            for source_id, mmsi, kind, observed_at, data, sentence_index in writes:
                self._write_message(
                    source_id, mmsi, kind, observed_at, data,
                    sentence_index, affected=affected,
                )
            # A late resolution backfills historical conflicts even when
            # it does not change the latest /vessels snapshots.
            self._refresh_conflicts(affected)
            # A late resolution likewise splits the jump segments of the
            # vessels whose tracks gained an older point.
            self._refresh_jump_segments(
                {(w[0], w[1]) for w in writes if w[2] == KIND_POSITION}
            )
            involved = sorted({(w[0], w[1]) for w in writes})
            snapshots = [
                self._to_snapshot(source_id, mmsi, self._fetch_rows(source_id, mmsi))
                for source_id, mmsi in involved
            ]
            resolution = json.dumps(
                {
                    "selections": selections,
                    "snapshots": [s.model_dump(mode="json") for s in snapshots],
                },
                separators=(",", ":"),
            )
            self._conn.execute(
                """
                UPDATE review_batch
                SET status = 'resolved', resolved_at = ?, resolution = ?
                WHERE batch_id = ?
                """,
                (resolved_at, resolution, batch_id),
            )
            return snapshots
