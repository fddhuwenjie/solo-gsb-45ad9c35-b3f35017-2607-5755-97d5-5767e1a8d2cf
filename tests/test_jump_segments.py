"""Integration tests for GET /jump-segments (suspected jump segment list)."""
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
    SnapshotStore,
    haversine_metres,
    normalize_observed_at,
)
from tests.conftest import (
    AMBIG_F1A,
    AMBIG_F1B,
    AMBIG_F2,
    SINGLE_FRAGMENT,
    SINGLE_PAYLOAD,
    VDO_SENTINELS,
    make_sentence,
)
from tests.test_conflicts import (
    BASE_LAT,
    BASE_LON,
    position_sentence,
)

T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"
T3 = "2026-10-05T13:00:00Z"

MMSI = 366999999
OTHER_MMSI = 367000000

# Same MMSI as SINGLE_FRAGMENT, ~287 m away: a slow, plausible move.
NEAR_POSITION = make_sentence("AIVDM,1,1,,A,15Mwqgh01DreF9HGD@44gkg40000,0")

# ~100 km north of the base point: ~54 knots over one hour -> a jump.
JUMP_LAT = BASE_LAT + 100_000 / 111_139
# ~200 km north of the base point: ~54 knots over two hours, ~108 over one.
FAR_JUMP_LAT = BASE_LAT + 200_000 / 111_139

JUMP_SENTENCE = position_sentence(MMSI, BASE_LON, JUMP_LAT)
FAR_JUMP_SENTENCE = position_sentence(MMSI, BASE_LON, FAR_JUMP_LAT)


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
    response = client.get("/jump-segments", params=params)
    return response.status_code, response.json()


def segment_keys(body: list[dict]) -> list[tuple]:
    return [
        (
            item["source_id"],
            item["mmsi"],
            item["start"]["observed_at"],
            item["end"]["observed_at"],
        )
        for item in body
    ]


# -- basic detection -------------------------------------------------------------


def test_fast_consecutive_points_record_one_jump_segment(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)
    assert status == 200

    status, body = jumps(client)
    assert status == 200
    assert len(body) == 1
    record = body[0]
    assert record["source_id"] == "rx-1"
    assert record["mmsi"] == MMSI
    assert record["start"]["observed_at"].startswith("2026-10-05T11:00:00")
    assert record["start"]["longitude"] == pytest.approx(BASE_LON, abs=1e-6)
    assert record["start"]["latitude"] == pytest.approx(BASE_LAT, abs=1e-6)
    assert record["end"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert record["end"]["longitude"] == pytest.approx(BASE_LON, abs=1e-6)
    assert record["end"]["latitude"] == pytest.approx(JUMP_LAT, abs=1e-4)
    distance = haversine_metres(
        record["start"]["longitude"], record["start"]["latitude"],
        record["end"]["longitude"], record["end"]["latitude"],
    )
    assert record["distance_m"] == pytest.approx(distance, abs=1e-3)
    assert record["average_speed_knots"] == pytest.approx(
        distance / 1852.0, rel=1e-3  # exactly one hour between the points
    )
    assert record["average_speed_knots"] > JUMP_SPEED_KNOTS


def test_slow_movement_records_no_segment(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    # ~287 m in one hour: far below 50 knots
    ingest(client, [one("rx-1", NEAR_POSITION)], T2)
    assert jumps(client)[1] == []


def test_speed_must_be_strictly_over_50_knots(client):
    """Exactly 50 knots is not a jump (strict ``>``); the pair must straddle it."""
    # 50 knots over one hour == 50 * 1852 m == 92600 m of great-circle distance
    below = position_sentence(MMSI, BASE_LON, BASE_LAT + 92_500 / 111_139)
    above = position_sentence(MMSI, BASE_LON, BASE_LAT + 92_700 / 111_139)
    decoded_below = decode_payload(below.split(",")[5], 0)
    decoded_above = decode_payload(above.split(",")[5], 0)
    knots_below = haversine_metres(
        BASE_LON, BASE_LAT, decoded_below.longitude, decoded_below.latitude
    ) / 1852.0
    knots_above = haversine_metres(
        BASE_LON, BASE_LAT, decoded_above.longitude, decoded_above.latitude
    ) / 1852.0
    assert knots_below < 50 < knots_above  # positions really straddle 50 kn

    # rx-1's pair stays below the threshold: no segment
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", below)], T2)
    # rx-2's pair crosses it: one segment
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", above)], T2)

    _, body = jumps(client)
    assert segment_keys(body) == [
        ("rx-2", MMSI, body[0]["start"]["observed_at"], body[0]["end"]["observed_at"])
    ]
    assert body[0]["source_id"] == "rx-2"
    assert body[0]["average_speed_knots"] > 50


def test_segments_are_per_source_and_need_two_points(client):
    # rx-1 has only the T1 point, rx-2 only the T2 point: no pair anywhere
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", JUMP_SENTENCE)], T2)
    assert jumps(client)[1] == []


# -- missing coordinates break adjacency ------------------------------------------


def test_null_coordinate_point_breaks_adjacency(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)      # real coordinates
    ingest(client, [one("rx-1", VDO_SENTINELS)], T2)        # longitude null
    ingest(client, [one("rx-1", FAR_JUMP_SENTENCE)], T3)    # ~200 km from T1

    # the gap point stays in the track ...
    track = client.get(
        "/tracks", params={"source_id": "rx-1", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 3
    assert track["points"][1]["data"]["longitude"] is None

    # ... but no segment is stitched across it: T1->T3 (~54 kn over 2 h)
    # would qualify, yet the null point in between breaks adjacency
    assert jumps(client)[1] == []


def test_null_coordinate_point_between_slow_pairs_lists_nothing(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", VDO_SENTINELS)], T2)
    ingest(client, [one("rx-1", NEAR_POSITION)], T3)
    assert jumps(client)[1] == []


# -- late writes split a segment ---------------------------------------------------


def test_late_observation_splits_existing_segment(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", FAR_JUMP_SENTENCE)], T3)  # T1->T3 ~54 kn
    _, body = jumps(client)
    assert [k[2:] for k in segment_keys(body)] == [
        (body[0]["start"]["observed_at"], body[0]["end"]["observed_at"])
    ]
    assert body[0]["start"]["observed_at"].startswith("2026-10-05T11:00:00")
    assert body[0]["end"]["observed_at"].startswith("2026-10-05T13:00:00")

    # a late T2 report lands between them: T1->T3 is replaced by T2->T3
    # (T1->T2 is only ~287 m and not listed)
    ingest(client, [one("rx-1", NEAR_POSITION)], T2)
    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["start"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert body[0]["end"]["observed_at"].startswith("2026-10-05T13:00:00")
    assert body[0]["average_speed_knots"] > 50


def test_late_review_resolution_splits_segment(client):
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_JUMP_SENTENCE)], T3)
    assert len(jumps(client, source_id="rx-2")[1]) == 1  # T1->T3

    # pending review batch at T2: a clear position plus an ambiguous pool
    created = client.post(
        "/reviews",
        json={
            "observed_at": T2,
            "sentences": [
                one("rx-2", NEAR_POSITION),
                one("rx-2", AMBIG_F1A),
                one("rx-2", AMBIG_F1B),
                one("rx-2", AMBIG_F2),
            ],
        },
    )
    assert created.status_code == 201
    batch_id = created.json()["batch_id"]

    # a pending batch changes nothing
    _, body = jumps(client, source_id="rx-2")
    assert len(body) == 1
    assert body[0]["end"]["observed_at"].startswith("2026-10-05T13:00:00")
    assert body[0]["start"]["observed_at"].startswith("2026-10-05T11:00:00")

    # resolving lands the T2 point: T1->T3 splits, only T2->T3 qualifies
    resolved = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 3]]}
    )
    assert resolved.status_code == 200
    _, body = jumps(client, source_id="rx-2")
    assert len(body) == 1
    assert body[0]["start"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert body[0]["end"]["observed_at"].startswith("2026-10-05T13:00:00")


# -- invalidation joins the neighbours ---------------------------------------------


def test_invalidating_middle_point_joins_neighbours(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", NEAR_POSITION)], T2)      # ~287 m from T1
    ingest(client, [one("rx-1", FAR_JUMP_SENTENCE)], T3)  # ~200 km from T1

    # only T2->T3 qualifies while the middle point is valid
    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["start"]["observed_at"].startswith("2026-10-05T12:00:00")

    # invalidating the middle point joins T1->T3 (~54 kn over two hours)
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
    assert body[0]["start"]["observed_at"].startswith("2026-10-05T11:00:00")
    assert body[0]["end"]["observed_at"].startswith("2026-10-05T13:00:00")
    assert body[0]["average_speed_knots"] > 50


def test_invalidation_that_yields_a_slow_pair_lists_nothing(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)   # T1->T2 is a jump
    ingest(client, [one("rx-1", NEAR_POSITION)], T3)   # ~200 km from T2: jump
    assert len(jumps(client)[1]) == 2

    # removing the middle point leaves T1->T3 (~287 m over two hours): slow
    client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": MMSI,
            "observed_at": T2,
            "reason": "misreport",
        },
    )
    assert jumps(client)[1] == []


# -- retransmissions and unchosen candidates ---------------------------------------


def test_same_instant_retransmission_changes_nothing(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)
    _, before = jumps(client)
    assert len(before) == 1

    # identical sentence at the same instant: no new track point, same list
    status, _ = ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)
    assert status == 200
    assert jumps(client)[1] == before

    # a different position at the same instant loses (first saved wins)
    ingest(client, [one("rx-1", NEAR_POSITION)], T2)
    assert jumps(client)[1] == before


def test_unchosen_candidates_add_no_segments(client):
    # rx-1 already has a jump T1->T2 for MMSI
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)

    # ambiguous batch for rx-2: one clear slow position + an ambiguous pool
    status, _ = ingest(
        client,
        [
            one("rx-2", SINGLE_FRAGMENT),
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
        T1,
    )
    assert status == 409
    # the unchosen candidates (MMSI 366000001/366000002) created no segments;
    # the list still holds exactly rx-1's jump
    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-1"
    assert jumps(client, source_id="rx-2")[1] == []
    assert jumps(client, mmsi=366000001)[1] == []
    assert jumps(client, mmsi=366000002)[1] == []


# -- ordering, filtering, persistence ----------------------------------------------


def _seed_ordering_tracks(client: TestClient) -> None:
    # (rx-1, MMSI): jumps ending at T2 and T3
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T3)
    # (rx-2, MMSI): jump ending at T2
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", JUMP_SENTENCE)], T2)
    # (rx-1, OTHER_MMSI): jump ending at T2
    other_base = position_sentence(OTHER_MMSI, BASE_LON, BASE_LAT)
    other_jump = position_sentence(OTHER_MMSI, BASE_LON, JUMP_LAT)
    ingest(client, [one("rx-1", other_base)], T1)
    ingest(client, [one("rx-1", other_jump)], T2)


def test_segments_ordered_by_end_time_mmsi_source(client):
    _seed_ordering_tracks(client)
    _, body = jumps(client)
    keys = [
        (
            datetime.fromisoformat(
                item["end"]["observed_at"].replace("Z", "+00:00")
            ),
            item["mmsi"],
            item["source_id"],
        )
        for item in body
    ]
    assert keys == sorted(keys)
    assert keys == [
        (datetime(2026, 10, 5, 12, tzinfo=timezone.utc), MMSI, "rx-1"),
        (datetime(2026, 10, 5, 12, tzinfo=timezone.utc), MMSI, "rx-2"),
        (datetime(2026, 10, 5, 12, tzinfo=timezone.utc), OTHER_MMSI, "rx-1"),
        (datetime(2026, 10, 5, 13, tzinfo=timezone.utc), MMSI, "rx-1"),
    ]


def test_filters_by_source_and_mmsi(client):
    _seed_ordering_tracks(client)

    _, body = jumps(client, source_id="rx-1")
    assert len(body) == 3
    assert all(item["source_id"] == "rx-1" for item in body)

    _, body = jumps(client, source_id="rx-2")
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-2"

    _, body = jumps(client, mmsi=OTHER_MMSI)
    assert len(body) == 1
    assert body[0]["mmsi"] == OTHER_MMSI

    _, body = jumps(client, source_id="rx-1", mmsi=MMSI)
    assert len(body) == 2
    assert all(
        item["source_id"] == "rx-1" and item["mmsi"] == MMSI for item in body
    )

    assert jumps(client, source_id="rx-9")[1] == []
    assert jumps(client, mmsi=999999999)[1] == []


def test_segments_persist_across_restart(client, monkeypatch):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)

    monkeypatch.setattr(main, "_store", None)  # simulate a container restart

    _, body = jumps(client)
    assert len(body) == 1
    assert body[0]["source_id"] == "rx-1"
    assert body[0]["average_speed_knots"] > 50


# -- migration from an existing database --------------------------------------------


def test_existing_database_backfills_segments_on_open(tmp_path):
    """A database whose tracks predate the feature lists them once opened."""
    path = str(tmp_path / "existing.db")
    store = SnapshotStore(path)
    observed_t1 = normalize_observed_at(datetime(2026, 10, 5, 11, tzinfo=timezone.utc))
    observed_t2 = normalize_observed_at(datetime(2026, 10, 5, 12, tzinfo=timezone.utc))
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    jump = decode_payload(JUMP_SENTENCE.split(",")[5], 0).model_dump()
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed_t1, base, 0)])
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed_t2, jump, 0)])
    store.close()

    # simulate the pre-feature database: tracks exist, the segment list does not
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE jump_segment")
    conn.commit()
    conn.close()

    reopened = SnapshotStore(path)
    segments = reopened.list_jump_segments()
    assert len(segments) == 1
    assert segments[0].source_id == "rx-1"
    assert segments[0].mmsi == MMSI
    assert segments[0].average_speed_knots > 50
    reopened.close()

    # reopening is idempotent: the list is rebuilt, not duplicated
    again = SnapshotStore(path)
    assert len(again.list_jump_segments()) == 1
    again.close()


def test_legacy_single_point_tracks_yield_no_segments(tmp_path):
    """An old vessel_state-only database seeds one track point per vessel."""
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
    message = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-1", MMSI, KIND_POSITION, "2026-10-05T12:00:00.000000Z",
         json.dumps(message)),
    )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    # one seeded track point per vessel cannot form a pair
    assert store.list_jump_segments() == []
    store.close()


# -- atomicity ----------------------------------------------------------------------


def test_segment_refresh_rolls_back_with_failed_batch(tmp_path):
    """A failing batch leaves no track point and no jump segment behind."""
    path = str(tmp_path / "atomic.db")
    store = SnapshotStore(path)
    observed_t1 = normalize_observed_at(datetime(2026, 10, 5, 11, tzinfo=timezone.utc))
    observed_t2 = normalize_observed_at(datetime(2026, 10, 5, 12, tzinfo=timezone.utc))
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    jump = decode_payload(JUMP_SENTENCE.split(",")[5], 0).model_dump()
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed_t1, base, 0)])

    good = ("rx-1", MMSI, KIND_POSITION, observed_t2, jump, 0)
    # a write whose payload cannot be serialised to JSON
    bad = ("rx-1", MMSI, KIND_POSITION, observed_t2, {"oops": object()}, 1)
    with pytest.raises(TypeError):
        store.apply_ingest([good, bad])

    # the track write, the snapshot upsert and the segment all rolled back
    assert len(store.get_track("rx-1", MMSI)) == 1
    assert store.list_jump_segments() == []
    store.close()


def test_failed_segment_refresh_rolls_back_invalidation(tmp_path, monkeypatch):
    """A failure while refreshing segments keeps the point valid."""
    store = SnapshotStore(str(tmp_path / "atomic.db"))
    observed = [
        normalize_observed_at(datetime(2026, 10, 5, hour, tzinfo=timezone.utc))
        for hour in (11, 12, 13)
    ]
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    near = decode_payload(NEAR_POSITION.split(",")[5], 0).model_dump()
    far = decode_payload(FAR_JUMP_SENTENCE.split(",")[5], 0).model_dump()
    store.apply_ingest(
        [
            ("rx-1", MMSI, KIND_POSITION, observed[0], base, 0),
            ("rx-1", MMSI, KIND_POSITION, observed[1], near, 0),
            ("rx-1", MMSI, KIND_POSITION, observed[2], far, 0),
        ]
    )
    assert len(store.list_jump_segments()) == 1  # T2 -> T3

    def boom(vessels):
        raise RuntimeError("simulated refresh failure")

    monkeypatch.setattr(store, "_refresh_jump_segments", boom)
    with pytest.raises(RuntimeError):
        store.invalidate_point(
            "rx-1", MMSI, observed[1], "r", "2026-10-06T00:00:00.000000Z"
        )

    # the marking rolled back together with the segment refresh
    assert len(store.get_track("rx-1", MMSI)) == 3
    segments = store.list_jump_segments()
    assert len(segments) == 1
    assert normalize_observed_at(segments[0].start.observed_at) == observed[1]
    store.close()


# -- read-only preview ---------------------------------------------------------------


def test_preview_leaves_no_segment_changes(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)

    response = client.post(
        "/ingest/preview",
        json={"observed_at": T2, "sentences": [one("rx-1", JUMP_SENTENCE)]},
    )
    assert response.status_code == 200
    body = response.json()
    # the preview report keeps its original shape (no segment fields)
    assert set(body) == {
        "observed_at",
        "summary",
        "results",
        "snapshot_changes",
        "new_track_points",
        "new_position_conflicts",
    }
    assert len(body["new_track_points"]) == 1

    # the dry run left no segment, track point or snapshot behind
    assert jumps(client)[1] == []
    track = client.get(
        "/tracks", params={"source_id": "rx-1", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 1

    # the real ingest of the same batch then produces the segment
    ingest(client, [one("rx-1", JUMP_SENTENCE)], T2)
    assert len(jumps(client)[1]) == 1
