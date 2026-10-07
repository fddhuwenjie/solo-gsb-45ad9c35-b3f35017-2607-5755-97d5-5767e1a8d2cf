"""Integration tests for GET /position-conflicts (multi-receiver conflict list)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ais import decode_payload
from app.storage import (
    KIND_POSITION,
    CONFLICT_DISTANCE_METRES,
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
    checksum,
)

T1 = "2026-10-05T11:00:00Z"
T2 = "2026-10-05T12:00:00Z"
T3 = "2026-10-05T13:00:00Z"

MMSI = 366999999
NORTHERN = 366000001
SOUTHERN = 366000002

# Coordinates of SINGLE_FRAGMENT: lon=-73.9835, lat=40.7484.
BASE_LON, BASE_LAT = -73.9835, 40.7484
# ~1.1 km from the base -> below the 1852 m threshold.
NEAR_LON, NEAR_LAT = -73.97, 40.7484
# ~2.4 km from the base -> above the threshold.
FAR_LON, FAR_LAT = -73.955, 40.7484
# ~7.0 km from the base and ~4.6 km from FAR -> pairwise conflicts everywhere.
FARTHER_LON, FARTHER_LAT = -73.9, 40.7484


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


def conflicts(
    client: TestClient, source_id: str | None = None, mmsi: int | None = None
):
    params = {}
    if source_id is not None:
        params["source_id"] = source_id
    if mmsi is not None:
        params["mmsi"] = mmsi
    response = client.get("/position-conflicts", params=params)
    return response.status_code, response.json()


# -- message 1 encoder ----------------------------------------------------------


def _ubits(value: int, length: int) -> str:
    return format(value, f"0{length}b")


def _sbits(value: int, length: int) -> str:
    if value < 0:
        value += 1 << length
    return format(value, f"0{length}b")


def position_sentence(
    mmsi: int,
    longitude: float,
    latitude: float,
    *,
    sog: float = 8.4,
    cog: float = 121.5,
    heading: int = 119,
    channel: str = "A",
    vdo: bool = False,
) -> str:
    """Build a self-consistent single-fragment type 1 sentence."""
    bits = (
        _ubits(1, 6) + _ubits(0, 2) + _ubits(mmsi, 30) + _ubits(0, 4)
        + _sbits(0, 8) + _ubits(round(sog * 10), 10) + _ubits(1, 1)
        + _sbits(round(longitude * 600000), 28)
        + _sbits(round(latitude * 600000), 27)
        + _ubits(round(cog * 10), 12) + _ubits(heading, 9) + _ubits(60, 6)
        + _ubits(0, 2) + _ubits(0, 3) + _ubits(0, 1) + _ubits(0, 19)
    )
    assert len(bits) == 168
    fill = (-len(bits)) % 6
    bits += "0" * fill
    chars = []
    for i in range(0, len(bits), 6):
        value = int(bits[i:i + 6], 2)
        chars.append(chr(value + 48) if value < 40 else chr(value + 56))
    payload = "".join(chars)
    tag = "AIVDO" if vdo else "AIVDM"
    body = f"{tag},1,1,,{channel},{payload},{fill}"
    return f"!{body}*{checksum(body)}"


FAR_SENTENCE = position_sentence(MMSI, FAR_LON, FAR_LAT)
NEAR_SENTENCE = position_sentence(MMSI, NEAR_LON, NEAR_LAT)
FARTHER_SENTENCE = position_sentence(MMSI, FARTHER_LON, FARTHER_LAT)


# -- basic conflict detection ----------------------------------------------------


def test_distant_sources_at_same_instant_record_one_conflict(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    status, body = ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert status == 200

    status, body = conflicts(client)
    assert status == 200
    assert len(body) == 1
    record = body[0]
    assert record["mmsi"] == MMSI
    assert record["observed_at"].startswith("2026-10-05T11:00:00")
    # canonical ordering of the source pair
    assert record["source_id_a"] == "rx-1"
    assert record["source_id_b"] == "rx-2"
    assert record["point_a"]["longitude"] == pytest.approx(BASE_LON, abs=1e-6)
    assert record["point_a"]["latitude"] == pytest.approx(BASE_LAT, abs=1e-6)
    assert record["point_b"]["longitude"] == pytest.approx(FAR_LON, abs=1e-6)
    assert record["point_b"]["latitude"] == pytest.approx(FAR_LAT, abs=1e-6)
    assert record["distance_m"] == pytest.approx(
        haversine_metres(BASE_LON, BASE_LAT, FAR_LON, FAR_LAT), abs=1e-3
    )
    assert record["distance_m"] > CONFLICT_DISTANCE_METRES


def test_source_pair_is_canonical_regardless_of_ingest_order(client):
    # the lexicographically later source arrives first
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["source_id_a"] == "rx-1"
    assert body[0]["source_id_b"] == "rx-2"
    assert body[0]["point_a"]["longitude"] == pytest.approx(BASE_LON, abs=1e-6)
    assert body[0]["point_b"]["longitude"] == pytest.approx(FAR_LON, abs=1e-6)


def test_close_positions_record_no_conflict(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    # ~1.1 km apart: below the strictly-greater-than-1852-m threshold
    ingest(client, [one("rx-2", NEAR_SENTENCE)], T1)

    _, body = conflicts(client)
    assert body == []


def test_distance_must_be_strictly_over_1852_metres(client):
    """One nautical mile itself is not a conflict (strict ``>``)."""
    # latitudes chosen ~1840 m and ~1865 m north of the base point
    below = position_sentence(MMSI, BASE_LON, BASE_LAT + 1840 / 111_139)
    above = position_sentence(MMSI, BASE_LON, BASE_LAT + 1865 / 111_139)
    decoded_below = decode_payload(below.split(",")[5], int(below.rsplit(",", 1)[-1][0]))
    decoded_above = decode_payload(above.split(",")[5], int(above.rsplit(",", 1)[-1][0]))
    d_below = haversine_metres(
        BASE_LON, BASE_LAT, decoded_below.longitude, decoded_below.latitude
    )
    d_above = haversine_metres(
        BASE_LON, BASE_LAT, decoded_above.longitude, decoded_above.latitude
    )
    assert d_below < 1852 < d_above  # positions really straddle the threshold

    # at T1 the two reports are closer than one nautical mile: no pair
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", below)], T1)
    assert conflicts(client)[1] == []

    # one second later rx-2 reports beyond the threshold: the pair appears
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", above)], T2)
    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["source_id_a"] == "rx-1"
    assert body[0]["source_id_b"] == "rx-2"
    assert body[0]["observed_at"].startswith("2026-10-05T12:00:00")
    assert body[0]["distance_m"] > 1852


def test_different_observed_at_means_no_conflict(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T2)
    assert conflicts(client)[1] == []


def test_different_mmsi_means_no_conflict(client):
    other = position_sentence(366000003, FAR_LON, FAR_LAT)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", other)], T1)
    assert conflicts(client)[1] == []


def test_null_coordinate_points_never_participate(client):
    # VDO_SENTINELS: longitude null (latitude present)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", VDO_SENTINELS)], T1)
    assert conflicts(client)[1] == []

    # the null-coordinate point exists in the track but cannot pair with anyone
    track = client.get(
        "/tracks", params={"source_id": "rx-2", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 1
    assert track["points"][0]["data"]["longitude"] is None


def test_same_source_never_pairs_with_itself(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    # same instant resubmission, even with another position: first saved wins
    ingest(client, [one("rx-1", FAR_SENTENCE)], T1)
    assert conflicts(client)[1] == []
    _, track = None, client.get(
        "/tracks", params={"source_id": "rx-1", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 1


# -- ambiguous ingest and pending reviews ----------------------------------------


def test_ambiguous_ingest_updates_conflicts_for_clear_decodes_only(client):
    # rx-1 already reports the far position at T1
    ingest(client, [one("rx-1", FAR_SENTENCE)], T1)

    # rx-2 sends one clear position (SINGLE) plus an ambiguous static pool
    status, body = ingest(
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
    # the clear decode produced the conflict with rx-1
    _, listed = conflicts(client)
    assert len(listed) == 1
    assert listed[0]["source_id_a"] == "rx-1"
    assert listed[0]["source_id_b"] == "rx-2"
    # unchosen candidates never landed anywhere
    assert client.get(
        "/tracks", params={"source_id": "rx-2", "mmsi": NORTHERN}
    ).status_code == 404
    assert client.get(
        "/tracks", params={"source_id": "rx-2", "mmsi": SOUTHERN}
    ).status_code == 404


def test_pending_review_adds_nothing_and_late_resolution_backfills(client):
    # rx-1 reports the far position at the older T1
    ingest(client, [one("rx-1", FAR_SENTENCE)], T1)
    # rx-2 already has a newer T2 snapshot (SINGLE position)
    ingest(client, [one("rx-2", SINGLE_FRAGMENT)], T2)

    # pending review at T1: clear SINGLE position for rx-2 + ambiguous pool
    created = client.post(
        "/reviews",
        json={
            "observed_at": T1,
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

    # a pending batch writes no points and therefore no conflicts
    status, track = None, client.get(
        "/tracks", params={"source_id": "rx-2", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 1  # only the T2 ingest
    assert track["points"][0]["observed_at"].startswith("2026-10-05T12:00:00")
    assert conflicts(client)[1] == []

    # late resolution: the old T1 point enters the track and the historical
    # conflict is backfilled even though the /vessels snapshot stays at T2
    resolved = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 3]]}
    )
    assert resolved.status_code == 200

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["observed_at"].startswith("2026-10-05T11:00:00")
    assert body[0]["source_id_a"] == "rx-1"
    assert body[0]["source_id_b"] == "rx-2"

    track = client.get(
        "/tracks", params={"source_id": "rx-2", "mmsi": MMSI}
    ).json()
    assert len(track["points"]) == 2  # late T1 + newer T2
    vessels = client.get(
        "/vessels", params={"source_id": "rx-2", "mmsi": MMSI}
    ).json()
    # the snapshot was not rolled back by the late resolution
    assert vessels[0]["position"]["observed_at"].startswith(
        "2026-10-05T12:00:00"
    )


def test_rejected_resolution_leaves_conflict_list_untouched(client):
    ingest(client, [one("rx-1", FAR_SENTENCE)], T1)
    batch_id = client.post(
        "/reviews",
        json={
            "observed_at": T1,
            "sentences": [
                one("rx-2", SINGLE_FRAGMENT),
                one("rx-2", AMBIG_F1A),
                one("rx-2", AMBIG_F1B),
                one("rx-2", AMBIG_F2),
            ],
        },
    ).json()["batch_id"]
    response = client.post(
        f"/reviews/{batch_id}/resolve", json={"selections": [[1, 2]]}  # forged
    )
    assert response.status_code == 422
    assert conflicts(client)[1] == []
    assert client.get(
        "/tracks", params={"source_id": "rx-2", "mmsi": MMSI}
    ).status_code == 404


# -- invalidation ----------------------------------------------------------------


def test_invalidation_removes_involved_pairs_and_keeps_others(client):
    # three sources at T1, all pairwise farther than 1852 m -> 3 pairs
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    ingest(client, [one("rx-3", FARTHER_SENTENCE)], T1)
    _, body = conflicts(client)
    assert {
        (r["source_id_a"], r["source_id_b"]) for r in body
    } == {("rx-1", "rx-2"), ("rx-1", "rx-3"), ("rx-2", "rx-3")}

    # invalidate rx-2's point: both pairs involving it disappear,
    # the rx-1/rx-3 pair survives
    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-2",
            "mmsi": MMSI,
            "observed_at": T1,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200
    _, body = conflicts(client)
    assert [(r["source_id_a"], r["source_id_b"]) for r in body] == [
        ("rx-1", "rx-3")
    ]

    # same-instant retransmission cannot restore the removed pairs
    status, _ = ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    assert status == 200
    _, body = conflicts(client)
    assert [(r["source_id_a"], r["source_id_b"]) for r in body] == [
        ("rx-1", "rx-3")
    ]

    # invalidating the other side removes the last pair
    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-3",
            "mmsi": MMSI,
            "observed_at": T1,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200
    assert conflicts(client)[1] == []


def test_invalidation_at_one_instant_keeps_conflicts_at_other_instants(client):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T2)
    ingest(client, [one("rx-3", FARTHER_SENTENCE)], T2)

    # two independent conflicts, one per instant
    _, body = conflicts(client)
    assert len(body) == 2
    assert [r["observed_at"].startswith("2026-10-05T11:00:00") for r in body] == [
        True,
        False,
    ]

    # removing rx-2's T1 point drops only the T1 pair
    response = client.post(
        "/tracks/invalidate",
        json={
            "source_id": "rx-2",
            "mmsi": MMSI,
            "observed_at": T1,
            "reason": "misreport",
        },
    )
    assert response.status_code == 200
    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["observed_at"].startswith("2026-10-05T12:00:00")
    assert body[0]["source_id_a"] == "rx-2"
    assert body[0]["source_id_b"] == "rx-3"


# -- ordering, filtering, persistence --------------------------------------------


def test_conflicts_ordered_by_time_mmsi_and_source_ids(client):
    later_mmsi = 367000000
    # T2 pair for the first MMSI
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T2)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T2)
    # T1 pair for the first MMSI (two sources -> checks inner ordering)
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-0", FARTHER_SENTENCE)], T1)
    # T1 pair for the other MMSI
    ingest(
        client,
        [one("rx-1", position_sentence(later_mmsi, BASE_LON, BASE_LAT))],
        T1,
    )
    ingest(
        client,
        [one("rx-2", position_sentence(later_mmsi, FAR_LON, FAR_LAT))],
        T1,
    )

    _, body = conflicts(client)
    keys = [
        (
            datetime.fromisoformat(r["observed_at"].replace("Z", "+00:00")),
            r["mmsi"],
            r["source_id_a"],
            r["source_id_b"],
        )
        for r in body
    ]
    assert keys == sorted(keys)
    assert keys == [
        (datetime(2026, 10, 5, 11, tzinfo=timezone.utc), MMSI, "rx-0", "rx-1"),
        (
            datetime(2026, 10, 5, 11, tzinfo=timezone.utc),
            later_mmsi,
            "rx-1",
            "rx-2",
        ),
        (datetime(2026, 10, 5, 12, tzinfo=timezone.utc), MMSI, "rx-1", "rx-2"),
    ]


def test_filters_by_mmsi_and_either_source(client):
    other_mmsi = 367000000
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)
    ingest(
        client,
        [one("rx-1", position_sentence(other_mmsi, BASE_LON, BASE_LAT))],
        T1,
    )
    ingest(
        client,
        [one("rx-2", position_sentence(other_mmsi, FAR_LON, FAR_LAT))],
        T1,
    )

    _, body = conflicts(client, mmsi=other_mmsi)
    assert {r["mmsi"] for r in body} == {other_mmsi}
    assert len(body) == 1

    # source filter matches either side of the pair
    _, body = conflicts(client, source_id="rx-1")
    assert len(body) == 2
    assert all(
        r["source_id_a"] == "rx-1" or r["source_id_b"] == "rx-1" for r in body
    )
    _, body = conflicts(client, source_id="rx-2")
    assert len(body) == 2
    assert all(
        r["source_id_a"] == "rx-2" or r["source_id_b"] == "rx-2" for r in body
    )
    assert conflicts(client, source_id="rx-9")[1] == []

    # both filters together
    _, body = conflicts(client, source_id="rx-1", mmsi=MMSI)
    assert len(body) == 1
    assert body[0]["mmsi"] == MMSI


def test_conflicts_persist_across_restart(client, monkeypatch):
    ingest(client, [one("rx-1", SINGLE_FRAGMENT)], T1)
    ingest(client, [one("rx-2", FAR_SENTENCE)], T1)

    monkeypatch.setattr(main, "_store", None)  # simulate a container restart

    _, body = conflicts(client)
    assert len(body) == 1
    assert body[0]["source_id_a"] == "rx-1"
    assert body[0]["source_id_b"] == "rx-2"
    assert body[0]["distance_m"] > 1852


# -- migration from an older database --------------------------------------------


def test_legacy_database_backfills_conflicts_on_open(tmp_path):
    """An old database holding only vessel_state rows surfaces conflicts."""
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
    observed = "2026-10-05T12:00:00.000000Z"
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    far = decode_payload(FAR_SENTENCE.split(",")[5], 0).model_dump()
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-1", MMSI, KIND_POSITION, observed, json.dumps(base)),
    )
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-2", MMSI, KIND_POSITION, observed, json.dumps(far)),
    )
    # a close third source: same instant, below the threshold
    near = decode_payload(NEAR_SENTENCE.split(",")[5], 0).model_dump()
    conn.execute(
        "INSERT INTO vessel_state VALUES (?, ?, ?, ?, ?)",
        ("rx-3", MMSI, KIND_POSITION, observed, json.dumps(near)),
    )
    conn.commit()
    conn.close()

    store = SnapshotStore(path)
    records = store.list_position_conflicts()
    assert len(records) == 1  # far pair only; close source adds nothing
    assert records[0].mmsi == MMSI
    assert records[0].source_id_a == "rx-1"
    assert records[0].source_id_b == "rx-2"
    assert records[0].distance_m > 1852
    store.close()

    # reopening is idempotent: no duplicated/missing rows
    reopened = SnapshotStore(path)
    again = reopened.list_position_conflicts()
    assert len(again) == 1
    assert again[0].source_id_a == "rx-1"
    assert again[0].source_id_b == "rx-2"
    reopened.close()


# -- atomicity -------------------------------------------------------------------


def test_conflict_refresh_commits_or_rolls_back_with_writes(tmp_path):
    """A failure before the conflict refresh leaves no partial update."""
    path = str(tmp_path / "atomic.db")
    store = SnapshotStore(path)
    observed = normalize_observed_at(
        datetime(2026, 10, 5, 11, 0, tzinfo=timezone.utc)
    )
    far = decode_payload(FAR_SENTENCE.split(",")[5], 0).model_dump()
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()

    # rx-1 far point already exists at T1
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed, far, 0)])
    assert len(store.list_position_conflicts()) == 0

    good = ("rx-2", MMSI, KIND_POSITION, observed, base, 1)
    bad = ("rx-2", MMSI, KIND_POSITION, observed, {"oops": object()}, 2)
    with pytest.raises(TypeError):
        store.apply_ingest([good, bad])

    # the track write, the snapshot upsert and the conflict refresh all rolled back
    assert store.get_track("rx-2", MMSI) == []
    assert store.get_snapshot("rx-2", MMSI) is None
    assert store.list_position_conflicts() == []
    store.close()


def test_failed_conflict_refresh_rolls_back_invalidation(
    tmp_path, monkeypatch
):
    """A failure while refreshing conflicts keeps the point valid."""
    store = SnapshotStore(str(tmp_path / "atomic.db"))
    observed = "2026-10-05T11:00:00.000000Z"
    far = decode_payload(FAR_SENTENCE.split(",")[5], 0).model_dump()
    base = decode_payload(SINGLE_PAYLOAD, 0).model_dump()
    store.apply_ingest([("rx-1", MMSI, KIND_POSITION, observed, far, 0)])
    store.apply_ingest([("rx-2", MMSI, KIND_POSITION, observed, base, 0)])
    assert len(store.list_position_conflicts()) == 1

    def boom(groups):
        raise RuntimeError("simulated refresh failure")

    monkeypatch.setattr(store, "_refresh_conflicts", boom)
    with pytest.raises(RuntimeError):
        store.invalidate_point(
            "rx-1", MMSI, observed, "r", "2026-10-06T00:00:00.000000Z"
        )

    # the marking and the conflict deletion both rolled back
    assert len(store.get_track("rx-1", MMSI)) == 1
    assert len(store.list_position_conflicts()) == 1
    store.close()
