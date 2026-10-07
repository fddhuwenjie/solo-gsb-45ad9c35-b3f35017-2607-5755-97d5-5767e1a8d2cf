"""Integration tests for POST /ingest/preview (read-only impact dry run)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ais import decode_payload
from app.storage import KIND_POSITION, SnapshotStore, normalize_observed_at
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
from tests.test_conflicts import (
    BASE_LAT,
    BASE_LON,
    FAR_LAT,
    FAR_LON,
    FAR_SENTENCE,
    MMSI,
    position_sentence,
)

T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"
T3 = "2026-10-05T13:00:00Z"

# Same MMSI as SINGLE_FRAGMENT but a different (older-track) longitude.
OTHER_POSITION = "!AIVDM,1,1,,A,15Mwqgh01DreF9HGD@44gkg40000,0*1F"


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


def preview(client: TestClient, sentences: list[dict], observed_at: str):
    response = client.post(
        "/ingest/preview", json={"observed_at": observed_at, "sentences": sentences}
    )
    return response.status_code, response.json()


# -- basic report shape ---------------------------------------------------------


def test_preview_new_vessel_lists_snapshot_change_and_track_point(client):
    status, body = preview(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    assert status == 200
    assert body["summary"]["decoded"] == 1
    assert body["results"][0]["status"] == "decoded"

    (change,) = body["snapshot_changes"]
    assert change["source_id"] == "rx-1"
    assert change["mmsi"] == MMSI
    assert change["before"] is None
    assert change["after"]["position"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert change["after"]["position"]["data"]["longitude"] == pytest.approx(-73.9835)
    assert change["after"]["static"] is None

    (point,) = body["new_track_points"]
    assert point["source_id"] == "rx-1"
    assert point["mmsi"] == MMSI
    assert point["point"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert point["point"]["data"]["mmsi"] == MMSI

    assert body["new_position_conflicts"] == []


def test_preview_static_message_changes_snapshot_without_track_point(client):
    status, body = preview(
        client, [one("rx-1", MULTI_FRAG2), one("rx-1", MULTI_FRAG1)], T2
    )
    assert status == 200
    (change,) = body["snapshot_changes"]
    assert change["before"] is None
    assert change["after"]["static"]["data"]["name"] == "MV CLAUDE"
    assert change["after"]["position"] is None
    assert body["new_track_points"] == []
    assert body["new_position_conflicts"] == []


def test_preview_writes_nothing_on_success(client):
    status, _ = preview(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    assert status == 200
    assert client.get("/vessels").json() == []
    assert (
        client.get("/tracks", params={"source_id": "rx-1", "mmsi": MMSI}).status_code
        == 404
    )
    assert client.get("/position-conflicts").json() == []
    # a second, independent preview sees the same empty before-state
    status, body = preview(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    assert status == 200
    assert len(body["snapshot_changes"]) == 1
    assert body["snapshot_changes"][0]["before"] is None
    assert len(body["new_track_points"]) == 1


def test_preview_incomplete_batch_reports_empty_diff_and_writes_nothing(client):
    status, body = preview(
        client, [one("rx-3", "!AIVDM,2,1,9,A,55Mwqgh2:N2UL@?7;?@mJ0<h5D@D000000000016?0N:<6Lf,0*53")], T2
    )
    assert status == 200
    assert body["results"][0]["status"] == "incomplete"
    assert body["snapshot_changes"] == []
    assert body["new_track_points"] == []
    assert body["new_position_conflicts"] == []
    assert client.get("/vessels").json() == []


def test_preview_missing_observed_at_rejected_with_422(client):
    response = client.post(
        "/ingest/preview", json={"sentences": [one("rx-1", SINGLE_FRAGMENT)]}
    )
    assert response.status_code == 422


# -- snapshot rules --------------------------------------------------------------


def test_preview_newer_batch_shows_before_and_after(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = preview(client, [one("rx-1", OTHER_POSITION)], T2)
    assert status == 200
    (change,) = body["snapshot_changes"]
    assert change["before"]["position"]["observed_at"].startswith("2026-10-05T11:00:00")
    assert change["before"]["position"]["data"]["longitude"] == pytest.approx(-73.9835)
    assert change["after"]["position"]["observed_at"].startswith("2026-10-05T12:00:00")
    assert change["after"]["position"]["data"]["longitude"] == pytest.approx(-73.980087)
    # same-instant track point already exists for T1, and T2 is new
    assert len(body["new_track_points"]) == 1


def test_preview_older_position_keeps_snapshot_but_still_adds_track_point(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    status, body = preview(client, [one("rx-1", OTHER_POSITION)], T1)
    assert status == 200
    # the latest snapshot is not rolled back -> no snapshot change reported
    assert body["snapshot_changes"] == []
    # ... but the late observation still enters the track
    (point,) = body["new_track_points"]
    assert point["source_id"] == "rx-1"
    assert point["point"]["observed_at"].startswith("2026-10-05T11:00:00")
    # the dry run really did not append it
    points = client.get(
        "/tracks", params={"source_id": "rx-1", "mmsi": MMSI}
    ).json()["points"]
    assert len(points) == 1


def test_preview_same_instant_first_saved_reports_no_change(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    status, body = preview(client, [one("rx-1", OTHER_POSITION)], T2)
    assert status == 200
    assert body["snapshot_changes"] == []
    assert body["new_track_points"] == []
    assert body["new_position_conflicts"] == []


def test_preview_null_coordinates_keep_null_in_after_snapshot(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = preview(client, [one("rx-1", VDO_SENTINELS)], T2)
    assert status == 200
    (change,) = body["snapshot_changes"]
    position = change["after"]["position"]["data"]
    assert position["longitude"] is None
    assert position["latitude"] == pytest.approx(40.7484)


def test_preview_ambiguous_batch_is_409_and_only_explicit_messages_count(client):
    status, body = preview(
        client,
        [
            one("rx-2", SINGLE_FRAGMENT),  # decodes unambiguously
            one("rx-2", AMBIG_F1A),
            one("rx-2", AMBIG_F1B),
            one("rx-2", AMBIG_F2),
        ],
        T2,
    )
    assert status == 409
    assert {r["status"] for r in body["results"]} == {"decoded", "ambiguous"}
    # only the unambiguous report contributes a change
    assert {c["mmsi"] for c in body["snapshot_changes"]} == {MMSI}
    assert {p["mmsi"] for p in body["new_track_points"]} == {MMSI}
    # nothing persisted despite the 409
    assert client.get("/vessels").json() == []
    assert client.get("/position-conflicts").json() == []


# -- conflict rules --------------------------------------------------------------


def test_preview_lists_new_same_instant_conflict_without_persisting(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = preview(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert status == 200
    (conflict,) = body["new_position_conflicts"]
    assert conflict["mmsi"] == MMSI
    assert conflict["observed_at"].startswith("2026-10-05T11:00:00")
    assert conflict["source_id_a"] == "rx-1"
    assert conflict["source_id_b"] == "rx-2"
    assert conflict["distance_m"] > 1852
    # dry run: the conflict is not stored
    assert client.get("/position-conflicts").json() == []


def test_preview_late_position_backfills_historical_conflict(client):
    # rx-1 is already at T1 and T3; rx-2 only reports at T3 (conflict at T3).
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T3)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T3)
    assert len(client.get("/position-conflicts").json()) == 1

    # A late rx-2 report at T1 would add the historical pair but cannot
    # change rx-2's latest snapshot (still T3).
    status, body = preview(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert status == 200
    assert body["snapshot_changes"] == []
    assert len(body["new_track_points"]) == 1
    (conflict,) = body["new_position_conflicts"]
    assert conflict["observed_at"].startswith("2026-10-05T11:00:00")

    # only the actual ingest materialises the backfilled conflict
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    stored = client.get("/position-conflicts").json()
    assert {c["observed_at"] for c in stored} == {
        "2026-10-05T11:00:00Z",
        "2026-10-05T13:00:00Z",
    }


def test_preview_existing_conflict_pair_is_not_listed_again(client):
    # the rx-1/rx-2 pair at T1 already exists
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    # a third source ~7 km away would add two *new* pairs only
    from tests.test_conflicts import FARTHER_SENTENCE

    status, body = preview(client, [one("rx-3", FARTHER_SENTENCE)], T1)
    assert status == 200
    pairs = {
        (c["source_id_a"], c["source_id_b"])
        for c in body["new_position_conflicts"]
    }
    assert pairs == {("rx-1", "rx-3"), ("rx-2", "rx-3")}


def test_preview_null_coordinates_create_no_conflict(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = preview(client, [one("rx-2", VDO_SENTINELS)], T1)
    assert status == 200
    assert len(body["new_track_points"]) == 1
    assert body["new_position_conflicts"] == []


# -- invalidated points -----------------------------------------------------------


def test_preview_does_not_revive_invalidated_point(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    invalidate = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": MMSI,
            "observed_at": T1,
            "reason": "false fix",
        },
    )
    assert invalidate.status_code == 200

    status, body = preview(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    assert status == 200
    assert body["snapshot_changes"] == []
    assert body["new_track_points"] == []
    assert body["new_position_conflicts"] == []
    # neither tracks nor snapshots reappear after the dry run
    assert client.get("/tracks", params={"source_id": "rx-1", "mmsi": MMSI}).json()["points"] == []
    assert client.get("/vessels").json() == []


# -- equivalence with the real /ingest ----------------------------------------------


def test_preview_changes_match_following_ingest_without_other_writes(client):
    # existing state: rx-1 at T1
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)

    batch = [one("rx-1", OTHER_POSITION), one("rx-2", FAR_SENTENCE)]
    status, preview_body = preview(client, batch, T2)
    assert status == 200

    # follow-up ingest with no intervening write
    status, ingest_body = ingest(client, batch, T2)
    assert status == 200

    # snapshots actually stored match the preview's after-values
    after_by_pair = {
        (c["source_id"], c["mmsi"]): c["after"]
        for c in preview_body["snapshot_changes"]
    }
    assert len(after_by_pair) == 2
    for snapshot in ingest_body["snapshots"]:
        key = (snapshot["source_id"], snapshot["mmsi"])
        assert key in after_by_pair
        assert snapshot == after_by_pair[key]

    # the previewed new points are now in the tracks, with no extras:
    # rx-1 keeps its prior T1 point and gains T2; rx-2 starts at T2.
    expected_points = {"rx-1": [T1, T2], "rx-2": [T2]}
    for source_id, expected in expected_points.items():
        points = client.get(
            "/tracks", params={"source_id": source_id, "mmsi": MMSI}
        ).json()["points"]
        assert [p["observed_at"] for p in points] == expected

    # and the stored conflict equals the previewed one
    (conflict,) = client.get("/position-conflicts").json()
    assert conflict == preview_body["new_position_conflicts"][0]

    # re-previewing the same batch is now a no-op
    status, body = preview(client, batch, T2)
    assert status == 200
    assert body["snapshot_changes"] == []
    assert body["new_track_points"] == []
    assert body["new_position_conflicts"] == []


def test_preview_rolls_back_when_pipeline_fails(tmp_path):
    """A failure inside the dry-run transaction leaves no partial write."""
    store = SnapshotStore(str(tmp_path / "preview.db"))
    observed = normalize_observed_at(
        datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    )
    data = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    writes = [("rx-1", MMSI, KIND_POSITION, observed, data, 0)]

    def fail_refresh(_groups):
        raise RuntimeError("simulated failure")

    store._refresh_conflicts = fail_refresh  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        store.preview_ingest(writes)
    del store._refresh_conflicts

    # rollback ran: nothing is stored and the store still works normally
    assert store.get_snapshot("rx-1", MMSI) is None
    assert store.list_position_conflicts() == []
    involved = store.apply_ingest(writes)
    assert involved == [("rx-1", MMSI)]
    snapshot = store.get_snapshot("rx-1", MMSI)
    assert snapshot is not None
    assert snapshot.position is not None
    store.close()


# -- large batches (many affected (MMSI, observed_at) groups) ---------------------

BULK_MMSIS = [200000000 + i for i in range(1000)]


def test_preview_large_batch_with_a_thousand_groups(client):
    """A batch forming ~1000 (MMSI, observed_at) groups still previews.

    Reading the affected groups used to stitch one OR clause per group into
    the before/after queries, which tripped SQLite's expression-depth limit
    (1000) for a legal bulk submission (<= 2000 sentences).
    """
    # pre-existing state:
    # - MMSI 366999999 stored at T1 by rx-1 and rx-2 (existing T1 conflict)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    # - BULK_MMSIS[1] already stored at T2: the batch re-sends it (no-op)
    ingest(client, [one("rx-1", position_sentence(BULK_MMSIS[1], BASE_LON, BASE_LAT))], T2)
    # - BULK_MMSIS[2] stored at T2 and invalidated: the batch cannot revive it
    ingest(client, [one("rx-1", position_sentence(BULK_MMSIS[2], BASE_LON, BASE_LAT))], T2)
    assert client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-1",
            "mmsi": BULK_MMSIS[2],
            "observed_at": T2,
            "reason": "misreport",
        },
    ).status_code == 200
    # - BULK_MMSIS[3] already has a newer T3 snapshot: the batch is late
    ingest(client, [one("rx-1", position_sentence(BULK_MMSIS[3], BASE_LON + 0.01, BASE_LAT))], T3)

    vessels_before = client.get("/vessels").json()
    conflicts_before = client.get("/position-conflicts").json()
    assert len(conflicts_before) == 1  # the T1 pair of MMSI 366999999

    # the bulk batch: 1000 distinct MMSIs from rx-1 (each its own group),
    # a second, far-away source for BULK_MMSIS[0] (one new conflict) and a
    # newer report for the pre-existing MMSI — 1002 sentences, within the
    # 2000-sentence cap
    sentences = [
        one("rx-1", position_sentence(mmsi, BASE_LON, BASE_LAT))
        for mmsi in BULK_MMSIS
    ]
    sentences.append(one("rx-2", position_sentence(BULK_MMSIS[0], FAR_LON, FAR_LAT)))
    sentences.append(one("rx-1", SINGLE_FRAGMENT))
    assert len(sentences) == 1002

    status, body = preview(client, sentences, T2)
    assert status == 200
    assert body["summary"]["decoded"] == 1002

    # snapshot changes: every bulk MMSI except the retransmitted [1], the
    # invalidated [2] and the late [3] (its newer T3 snapshot stays), plus
    # the second source of BULK_MMSIS[0] and the newer report of MMSI
    changed = {(c["source_id"], c["mmsi"]) for c in body["snapshot_changes"]}
    expected = {
        ("rx-1", mmsi)
        for mmsi in BULK_MMSIS
        if mmsi not in (BULK_MMSIS[1], BULK_MMSIS[2], BULK_MMSIS[3])
    } | {("rx-2", BULK_MMSIS[0]), ("rx-1", MMSI)}
    assert changed == expected

    by_pair = {(c["source_id"], c["mmsi"]): c for c in body["snapshot_changes"]}
    assert by_pair[("rx-1", BULK_MMSIS[4])]["before"] is None
    assert by_pair[("rx-1", MMSI)]["before"]["position"][
        "observed_at"
    ].startswith("2026-10-05T11:00:00")
    assert by_pair[("rx-1", MMSI)]["after"]["position"][
        "observed_at"
    ].startswith("2026-10-05T12:00:00")

    # new track points: the same pairs, plus the late point of BULK_MMSIS[3]
    new_points = {(p["source_id"], p["mmsi"]) for p in body["new_track_points"]}
    assert new_points == expected | {("rx-1", BULK_MMSIS[3])}
    for point in body["new_track_points"]:
        assert point["point"]["observed_at"].startswith("2026-10-05T12:00:00")

    # exactly one new conflict: rx-1/rx-2 for BULK_MMSIS[0] at T2 — the
    # pre-existing T1 conflict of MMSI 366999999 is not listed again
    (conflict,) = body["new_position_conflicts"]
    assert conflict["mmsi"] == BULK_MMSIS[0]
    assert conflict["observed_at"].startswith("2026-10-05T12:00:00")
    assert (conflict["source_id_a"], conflict["source_id_b"]) == ("rx-1", "rx-2")

    # the dry run persisted nothing at all
    assert client.get("/vessels").json() == vessels_before
    assert client.get("/position-conflicts").json() == conflicts_before

    # a repeated preview (group staging reused after the rollback) is identical
    status, again = preview(client, sentences, T2)
    assert status == 200
    assert again["snapshot_changes"] == body["snapshot_changes"]
    assert again["new_track_points"] == body["new_track_points"]
    assert again["new_position_conflicts"] == body["new_position_conflicts"]

    # with no intervening write, the real ingest applies exactly the preview
    status, applied = ingest(client, sentences, T2)
    assert status == 200
    after_by_pair = {
        (c["source_id"], c["mmsi"]): c["after"] for c in body["snapshot_changes"]
    }
    unchanged = {("rx-1", BULK_MMSIS[1]), ("rx-1", BULK_MMSIS[3])}
    for snapshot in applied["snapshots"]:
        key = (snapshot["source_id"], snapshot["mmsi"])
        if key in after_by_pair:
            assert snapshot == after_by_pair[key]
        else:  # retransmission / late point: stored snapshot untouched
            assert key in unchanged
    stored = client.get("/position-conflicts").json()
    assert len(stored) == 2  # pre-existing T1 pair + the previewed T2 pair
    assert conflict in stored

    # the late point entered the track without rolling back the T3 snapshot
    points = client.get(
        "/tracks", params={"source_id": "rx-1", "mmsi": BULK_MMSIS[3]}
    ).json()["points"]
    assert [p["observed_at"] for p in points] == [
        "2026-10-05T12:00:00Z",
        "2026-10-05T13:00:00Z",
    ]
    # the invalidated point stayed hidden and was not re-activated
    assert (
        client.get("/tracks", params={"source_id": "rx-1", "mmsi": BULK_MMSIS[2]})
        .json()["points"]
        == []
    )

    # re-previewing the applied batch is now a no-op
    status, noop = preview(client, sentences, T2)
    assert status == 200
    assert noop["snapshot_changes"] == []
    assert noop["new_track_points"] == []
    assert noop["new_position_conflicts"] == []
