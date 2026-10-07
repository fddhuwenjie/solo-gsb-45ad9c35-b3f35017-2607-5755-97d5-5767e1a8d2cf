"""Integration tests for GET /track-jumps (suspected jump segments)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ais import decode_payload
from app.storage import (
    JUMP_SPEED_KNOTS,
    KIND_POSITION,
    METRES_PER_NAUTICAL_MILE,
    SnapshotStore,
    haversine_metres,
    normalize_observed_at,
)
from tests.conftest import (
    AMBIG_F1A,
    AMBIG_F1B,
    AMBIG_F2,
    MULTI_FRAG1,
    MULTI_FRAG2,
    SINGLE_FRAGMENT,
    SINGLE_PAYLOAD,
    VDO_SENTINELS,
)
from tests.test_conflicts import position_sentence

T0 = "2026-10-05T10:00:00Z"
T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"
T3 = "2026-10-05T13:00:00Z"

MMSI = 366999999
NORTHERN = 366000001  # AMBIG_F1A + AMBIG_F2 -> "NORTHERN STAR"
OTHER_MMSI = 367000000

# Coordinates of SINGLE_FRAGMENT: lon=-73.9835, lat=40.7484.
BASE_LON, BASE_LAT = -73.9835, 40.7484
# ~111.2 km north of the base: a one-hour segment averages ~60 kn (> 50).
HOP_LAT = BASE_LAT + 1.0
# ~189 km north: ~51 kn over two hours, ~102 kn over one hour.
LEAP_LAT = BASE_LAT + 1.7
# ~222.4 km north of the base: ~60 kn over two hours.
DOUBLE_HOP_LAT = BASE_LAT + 2.0

HOP_SENTENCE = position_sentence(MMSI, BASE_LON, HOP_LAT)
LEAP_SENTENCE = position_sentence(MMSI, BASE_LON, LEAP_LAT)
DOUBLE_HOP_SENTENCE = position_sentence(MMSI, BASE_LON, DOUBLE_HOP_LAT)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by a fresh snapshot database per test."""
    monkeypatch.setenv("AIS_DB_PATH", str(tmp_path / "snapshots.db"))
    monkeypatch.setattr(main, "_store", None)
    yield TestClient(main.app)
    monkeypatch.setattr(main, "_store", None)


def one(source_id: str, sentence: str) -> dict:
    return {"source_id": source_id, "sentence": sentence}


def ingest(client: TestClient, sentences: list[dict], observed_at: str):
    response = client.post(
        "/ingest", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


def jumps(
    client: TestClient, source_id: str | None = None, mmsi: int | None = None
):
    params = {}
    if source_id is not None:
        params["source_id"] = source_id
    if mmsi is not None:
        params["mmsi"] = mmsi
    response = client.get("/track-jumps", params=params)
    return response.status_code, response.json()


# -- basic jump detection ----------------------------------------------------


def test_segment_over_50_knots_is_listed(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    assert status == 200

    status, body = jumps(client)
    assert status == 200
    assert len(body) == 1
    record = body[0]
    assert record["source_id"] == "rx-1"
    assert record["mmsi"] == MMSI
    assert record["start_observed_at"].startswith("2026-10-05T11:00:00")
    assert record["end_observed_at"].startswith("2026-10-05T12:00:00")
    assert record["start_point"]["longitude"] == pytest.approx(BASE_LON, abs=1e-6)
    assert record["start_point"]["latitude"] == pytest.approx(BASE_LAT, abs=1e-6)
    assert record["end_point"]["longitude"] == pytest.approx(BASE_LON, abs=1e-6)
    assert record["end_point"]["latitude"] == pytest.approx(HOP_LAT, abs=1e-6)
    distance = haversine_metres(BASE_LON, BASE_LAT, BASE_LON, HOP_LAT)
    assert record["distance_m"] == pytest.approx(distance, abs=1e-3)
    # one hour elapsed: knots == nautical miles covered
    assert record["speed_knots"] == pytest.approx(
        distance / METRES_PER_NAUTICAL_MILE, abs=1e-3
    )
    assert record["speed_knots"] > JUMP_SPEED_KNOTS


def test_slower_segment_is_not_listed(client):
    # ~44.5 km in one hour -> ~24 kn: below the threshold
    near = position_sentence(MMSI, BASE_LON, BASE_LAT + 0.4)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", near)], T2)
    assert jumps(client)[1] == []


def test_speed_must_be_strictly_over_50_knots(client):
    """Exactly 50 knots is not a jump (strict ``>``)."""
    # latitudes chosen ~92,578 m and ~92,633 m north of the base point;
    # the 50-knot threshold over one hour is exactly 92,600 m
    below = position_sentence(MMSI, BASE_LON, BASE_LAT + 0.8325)
    above = position_sentence(MMSI, BASE_LON, BASE_LAT + 0.8330)
    decoded_below = decode_payload(below.split(",")[5], 0)
    decoded_above = decode_payload(above.split(",")[5], 0)
    d_below = haversine_metres(
        BASE_LON, BASE_LAT, decoded_below.longitude, decoded_below.latitude
    )
    d_above = haversine_metres(
        BASE_LON, BASE_LAT, decoded_above.longitude, decoded_above.latitude
    )
    # positions really straddle the 50-knot threshold for a one-hour gap
    assert d_below / METRES_PER_NAUTICAL_MILE < 50
    assert d_above / METRES_PER_NAUTICAL_MILE > 50

    # rx-1's one-hour segment stays just below 50 knots: no record
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", below)], T2)
    assert jumps(client)[1] == []

    # rx-2's one-hour segment crosses the threshold: the record appears
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", above)], T2)
    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-2"
    assert body[0]["end_observed_at"].startswith("2026-10-05T12:00:00")
    assert body[0]["speed_knots"] > 50


def test_segments_never_span_sources(client):
    # same MMSI but different sources: each source has its own track
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", HOP_SENTENCE)], T2)
    assert jumps(client)[1] == []


def test_static_only_vessel_has_no_jumps(client):
    ingest(client, [one("rx-1", MULTI_FRAG1), one("rx-1", MULTI_FRAG2)], T1)
    assert jumps(client)[1] == []


# -- missing coordinates break the chain --------------------------------------


def test_null_coordinate_point_breaks_the_chain(client):
    # VDO_SENTINELS: longitude null (latitude present); without the break a
    # stitched T1->T3 segment would average ~51 kn and show up as a jump
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", VDO_SENTINELS)], T2)
    ingest(client, [one("rx-1", LEAP_SENTENCE)], T3)

    assert jumps(client)[1] == []
    # the missing-data point stays in the track
    track = client.get(
        "/tracks", params={"source_id": "rx-1", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 3
    assert track["points"][1]["data"]["longitude"] is None


# -- late writes split, invalidation joins ------------------------------------


def test_late_ingest_splits_existing_segment(client):
    # T1 -> T3 averages ~51 kn over two hours: one jump segment
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", LEAP_SENTENCE)], T3)
    _, body = jumps(client)
    assert [(r["start_observed_at"], r["end_observed_at"]) for r in body] == [
        ("2026-10-05T11:00:00Z", "2026-10-05T13:00:00Z")
    ]

    # a late T2 report ~1.1 km north of the base splits the segment:
    # T1->T2 is ~0.6 kn (no jump), T2->T3 is ~101 kn (jump)
    near_base = position_sentence(MMSI, BASE_LON, BASE_LAT + 0.01)
    ingest(client, [one("rx-1", near_base)], T2)
    _, body = jumps(client)
    assert len(body) == 1
    record = body[0]
    assert record["start_observed_at"].startswith("2026-10-05T12:00:00")
    assert record["end_observed_at"].startswith("2026-10-05T13:00:00")
    assert record["start_point"]["latitude"] == pytest.approx(BASE_LAT + 0.01, abs=1e-6)
    assert record["end_point"]["latitude"] == pytest.approx(LEAP_LAT, abs=1e-6)


def test_review_resolution_splits_existing_segment(client):
    # rx-2 has a T1 -> T3 jump segment
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", LEAP_SENTENCE)], T3)
    assert len(jumps(client)[1]) == 1

    # a review batch observed at T2 carries the base position again
    created = client.post(
        "/reviews",
        json={
            "observed_at": T2,
            "sentences": [
                one("rx-2", SINGLE_FRAGMENT),
                one("rx-2", AMBIG_F1A),
                one("rx-2", AMBIG_F1B),
                one("rx-2", AMBIG_F2),
            ],
        },
    )
    assert created.status_code == 201
    batch_id = created.json()["batch_id"]
    # pending: nothing changes
    assert len(jumps(client)[1]) == 1

    resolved = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 3]]}
    )
    assert resolved.status_code == 200
    # the T2 point splits the segment; only T2 -> T3 (~102 kn) remains
    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-2"
    assert body[0]["start_observed_at"].startswith("2026-10-05T12:00:00")
    assert body[0]["end_observed_at"].startswith("2026-10-05T13:00:00")


def test_invalidation_joins_the_neighbours(client):
    # two consecutive ~60 kn hops: T1 -> T2 -> T3
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    ingest(client, [one("rx-1", DOUBLE_HOP_SENTENCE)], T3)
    _, body = jumps(client)
    assert [(r["start_observed_at"], r["end_observed_at"]) for r in body] == [
        ("2026-10-05T11:00:00Z", "2026-10-05T12:00:00Z"),
        ("2026-10-05T12:00:00Z", "2026-10-05T13:00:00Z"),
    ]

    # invalidating the middle point joins the sides into one T1 -> T3 segment
    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": MMSI,
            "observed_at": T2,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200
    _, body = jumps(client)
    assert len(body) == 1
    record = body[0]
    assert record["start_observed_at"].startswith("2026-10-05T11:00:00")
    assert record["end_observed_at"].startswith("2026-10-05T13:00:00")
    assert record["start_point"]["latitude"] == pytest.approx(BASE_LAT, abs=1e-6)
    assert record["end_point"]["latitude"] == pytest.approx(DOUBLE_HOP_LAT, abs=1e-6)
    distance = haversine_metres(BASE_LON, BASE_LAT, BASE_LON, DOUBLE_HOP_LAT)
    assert record["distance_m"] == pytest.approx(distance, abs=1e-3)
    # two hours elapsed: ~60 kn, still a jump
    assert record["speed_knots"] == pytest.approx(
        distance / METRES_PER_NAUTICAL_MILE / 2, abs=1e-3
    )


def test_invalidation_join_that_is_not_a_jump_empties_the_list(client):
    # T1 -> T2 is a ~60 kn jump; T2 -> T3 is a ~0.3 kn drift (no jump)
    near_hop = position_sentence(MMSI, BASE_LON, HOP_LAT + 0.005)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    ingest(client, [one("rx-1", near_hop)], T3)
    assert len(jumps(client)[1]) == 1

    # the joined T1 -> T3 segment averages ~30 kn: the list becomes empty
    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": MMSI,
            "observed_at": T2,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200
    assert jumps(client)[1] == []


# -- writes that must not change the list --------------------------------------


def test_same_instant_retransmission_changes_nothing(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    _, before = jumps(client)
    assert len(before) == 1

    # the same sentence again at the same instant: no new track point
    status, _ = ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    assert status == 200
    assert jumps(client)[1] == before

    # a different position at the same instant loses (first saved wins)
    ingest(client, [one("rx-1", LEAP_SENTENCE)], T2)
    _, body = jumps(client)
    assert body == before
    assert body[0]["end_point"]["latitude"] == pytest.approx(HOP_LAT, abs=1e-6)


def test_pending_review_and_unchosen_candidates_add_nothing(client):
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)

    # a pending review batch writes no track points and no jump segments
    created = client.post(
        "/reviews",
        json={
            "observed_at": T2,
            "sentences": [
                one("rx-2", HOP_SENTENCE),
                one("rx-2", AMBIG_F1A),
                one("rx-2", AMBIG_F1B),
                one("rx-2", AMBIG_F2),
            ],
        },
    )
    assert created.status_code == 201
    batch_id = created.json()["batch_id"]
    assert jumps(client)[1] == []

    # resolution writes the T2 hop point: the jump appears for rx-2 only;
    # the selected static candidate and the unchosen one add no segments
    resolved = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 3]]}
    )
    assert resolved.status_code == 200
    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-2"
    assert body[0]["mmsi"] == MMSI
    assert jumps(client, mmsi=NORTHERN)[1] == []


def test_ambiguous_ingest_unchosen_candidates_add_nothing(client):
    status, _ = ingest(
        client,
        [one("rx-2", AMBIG_F1A), one("rx-2", AMBIG_F1B), one("rx-2", AMBIG_F2)],
        T2,
    )
    assert status == 409
    assert jumps(client)[1] == []


def test_rejected_resolution_leaves_jump_list_untouched(client):
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    batch_id = client.post(
        "/reviews",
        json={
            "observed_at": T2,
            "sentences": [
                one("rx-2", HOP_SENTENCE),
                one("rx-2", AMBIG_F1A),
                one("rx-2", AMBIG_F1B),
                one("rx-2", AMBIG_F2),
            ],
        },
    ).json()["batch_id"]
    # forged candidate -> whole resolution rejected
    response = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 2]]}
    )
    assert response.status_code == 422
    assert jumps(client)[1] == []


# -- ordering, filtering, persistence ------------------------------------------


def test_jumps_ordered_by_end_time_mmsi_and_source(client):
    # rx-2 reports MMSI at T0 -> T1 (jump ends T1)
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T0)
    ingest(client, [one("rx-2", HOP_SENTENCE)], T1)
    # rx-1 reports another MMSI at T0 -> T1 (jump ends T1)
    ingest(client, [one("rx-1", position_sentence(OTHER_MMSI, BASE_LON, BASE_LAT))], T0)
    ingest(client, [one("rx-1", position_sentence(OTHER_MMSI, BASE_LON, HOP_LAT))], T1)
    # rx-1 reports MMSI at T1 -> T3 (jump ends T3)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", LEAP_SENTENCE)], T3)

    _, body = jumps(client)
    keys = [
        (
            datetime.fromisoformat(r["end_observed_at"].replace("Z", "+00:00")),
            r["mmsi"],
            r["source_id"],
        )
        for r in body
    ]
    assert keys == sorted(keys)
    assert keys == [
        (datetime(2026, 10, 5, 11, tzinfo=timezone.utc), MMSI, "rx-2"),
        (datetime(2026, 10, 5, 11, tzinfo=timezone.utc), OTHER_MMSI, "rx-1"),
        (datetime(2026, 10, 5, 13, tzinfo=timezone.utc), MMSI, "rx-1"),
    ]


def test_filters_by_source_and_mmsi(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", HOP_SENTENCE)], T2)
    ingest(client, [one("rx-1", position_sentence(OTHER_MMSI, BASE_LON, BASE_LAT))], T1)
    ingest(client, [one("rx-1", position_sentence(OTHER_MMSI, BASE_LON, HOP_LAT))], T2)

    _, body = jumps(client)
    assert len(body) == 3

    _, body = jumps(client, source_id="rx-1")
    assert {(r["source_id"], r["mmsi"]) for r in body} == {
        ("rx-1", MMSI),
        ("rx-1", OTHER_MMSI),
    }
    _, body = jumps(client, mmsi=MMSI)
    assert {(r["source_id"], r["mmsi"]) for r in body} == {
        ("rx-1", MMSI),
        ("rx-2", MMSI),
    }
    _, body = jumps(client, source_id="rx-1", mmsi=MMSI)
    assert [(r["source_id"], r["mmsi"]) for r in body] == [("rx-1", MMSI)]
    assert jumps(client, source_id="rx-9")[1] == []
    assert jumps(client, mmsi=999999999)[1] == []


def test_jumps_persist_across_restart(client, monkeypatch):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", HOP_SENTENCE)], T2)

    monkeypatch.setattr(main, "_store", None)  # simulate a container restart

    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-1"
    assert body[0]["speed_knots"] > 50


# -- migration from an older database ------------------------------------------


def _legacy_track_database(path: str) -> None:
    """Build a database the way the pre-invalidation release did."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vessel_state (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL, kind TEXT NOT NULL,
            observed_at TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, kind)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE vessel_track (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL,
            observed_at TEXT NOT NULL, sentence_index INTEGER NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, observed_at)
        ) WITHOUT ROWID
        """
    )
    base = json.dumps(decode_payload(SINGLE_PAYLOAD, 0).model_dump())
    hop = json.dumps(decode_payload(HOP_SENTENCE.split(",")[5], 0).model_dump())
    conn.execute(
        "INSERT INTO vessel_track VALUES (?, ?, ?, ?, ?)",
        ("rx-1", MMSI, "2026-10-05T11:00:00.000000Z", 0, base),
    )
    conn.execute(
        "INSERT INTO vessel_track VALUES (?, ?, ?, ?, ?)",
        ("rx-1", MMSI, "2026-10-05T12:00:00.000000Z", 0, hop),
    )
    conn.commit()
    conn.close()


def test_legacy_database_backfills_jumps_on_open(tmp_path):
    """Tracks already stored in an older database surface their jump segments."""
    path = str(tmp_path / "legacy.db")
    _legacy_track_database(path)

    store = SnapshotStore(path)
    records = store.list_track_jumps()
    assert len(records) == 1
    record = records[0]
    assert record.source_id == "rx-1"
    assert record.mmsi == MMSI
    assert record.start_observed_at == datetime(2026, 10, 5, 11, tzinfo=timezone.utc)
    assert record.end_observed_at == datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    assert record.speed_knots > 50
    store.close()

    # reopening is idempotent: no duplicated/missing rows
    reopened = SnapshotStore(path)
    again = reopened.list_track_jumps()
    assert len(again) == 1
    assert again[0].start_observed_at == record.start_observed_at
    reopened.close()


def test_legacy_snapshot_only_database_has_single_points_and_no_jumps(tmp_path):
    """A vessel_state-only database seeds one track point per vessel: no pairs."""
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE vessel_state (
            source_id TEXT NOT NULL, mmsi INTEGER NOT NULL, kind TEXT NOT NULL,
            observed_at TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (source_id, mmsi, kind)
        )
        """
    )
    message = json.dumps(decode_payload(SINGLE_PAYLOAD, 0).model_dump())
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-1", MMSI, KIND_POSITION, "2026-10-05T12:00:00.000000Z", message),
    )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    assert len(store.get_track("rx-1", MMSI)) == 1  # one seeded start point
    assert store.list_track_jumps() == []
    store.close()


# -- atomicity ------------------------------------------------------------------


def test_jump_refresh_commits_or_rolls_back_with_writes(tmp_path):
    """A failure mid-batch rolls back track points, snapshots and segments."""
    path = str(tmp_path / "atomic.db")
    store = SnapshotStore(path)
    observed1 = normalize_observed_at(datetime(2026, 10, 5, 11, 0, tzinfo=timezone.utc))
    observed2 = normalize_observed_at(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc))
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    hop = decode_payload(HOP_SENTENCE.split(",")[5], 0).model_dump()
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed1, base, 0)])

    good = ("rx-1", MMSI, KIND_POSITION, observed2, hop, 1)
    # a write whose payload cannot be serialised to JSON
    bad = ("rx-1", MMSI, KIND_POSITION, observed2, {"oops": object()}, 2)
    with pytest.raises(TypeError):
        store.apply_ingest([good, bad])

    # the jump the good write would have created rolled back with everything
    assert len(store.get_track("rx-1", MMSI)) == 1
    assert store.list_track_jumps() == []
    store.close()


def test_failed_jump_refresh_rolls_back_invalidation(tmp_path, monkeypatch):
    """A failure while refreshing jumps keeps the point valid and the list."""
    store = SnapshotStore(str(tmp_path / "atomic.db"))
    observed1 = "2026-10-05T11:00:00.000000Z"
    observed2 = "2026-10-05T12:00:00.000000Z"
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    hop = decode_payload(HOP_SENTENCE.split(",")[5], 0).model_dump()
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed1, base, 0)])
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed2, hop, 0)])
    assert len(store.list_track_jumps()) == 1

    def boom(vessels):
        raise RuntimeError("simulated refresh failure")

    monkeypatch.setattr(store, "_refresh_jumps", boom)
    with pytest.raises(RuntimeError):
        store.invalidate_point(
            "rx-1", MMSI, observed2, "r", "2026-10-06T00:00:00.000000Z"
        )

    # the marking and the segment deletion both rolled back
    assert len(store.get_track("rx-1", MMSI)) == 2
    assert len(store.list_track_jumps()) == 1
    store.close()


def test_upsert_if_newer_refreshes_jumps(tmp_path):
    """The single-write store API maintains the jump list as well."""
    store = SnapshotStore(str(tmp_path / "store.db"))
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    hop = decode_payload(HOP_SENTENCE.split(",")[5], 0).model_dump()
    store.upsert_if_newer(
        "rx-1", MMSI, KIND_POSITION, "2026-10-05T11:00:00.000000Z", base
    )
    assert store.list_track_jumps() == []
    store.upsert_if_newer(
        "rx-1", MMSI, KIND_POSITION, "2026-10-05T12:00:00.000000Z", hop
    )
    records = store.list_track_jumps()
    assert len(records) == 1
    assert records[0].speed_knots > 50
    store.close()


# -- read-only preview ----------------------------------------------------------


def test_preview_leaves_jump_list_untouched(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)

    response = client.post(
        "/ingest/preview",
        json={"observed_at": T2, "sentences": [one("rx-1", HOP_SENTENCE)]},
    )
    assert response.status_code == 200
    # the preview contract is unchanged: no jump-segment field leaks in
    assert set(response.json()) == {
        "observed_at",
        "summary",
        "results",
        "snapshot_changes",
        "new_track_points",
        "new_position_conflicts",
    }
    # the dry run left no jump segment behind
    assert jumps(client)[1] == []

    # the real ingest materialises the segment
    ingest(client, [one("rx-1", HOP_SENTENCE)], T2)
    assert len(jumps(client)[1]) == 1


def test_preview_rolls_back_when_jump_refresh_fails(tmp_path, monkeypatch):
    """A failing jump refresh inside the dry run leaves no partial write."""
    store = SnapshotStore(str(tmp_path / "preview.db"))
    observed1 = normalize_observed_at(datetime(2026, 10, 5, 11, 0, tzinfo=timezone.utc))
    observed2 = normalize_observed_at(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc))
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    hop = decode_payload(HOP_SENTENCE.split(",")[5], 0).model_dump()
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed1, base, 0)])

    def boom(vessels):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(store, "_refresh_jumps", boom)
    with pytest.raises(RuntimeError):
        store.preview_ingest([("rx-1", MMSI, KIND_POSITION, observed2, hop, 0)])

    # rollback ran: the point was not stored and no segment was recorded
    assert len(store.get_track("rx-1", MMSI)) == 1
    assert store.list_track_jumps() == []
    store.close()
