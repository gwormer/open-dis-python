"""Regression tests for the standalone connector; no listener or wall-clock waits."""

import math
import sys
from unittest.mock import Mock

import pytest

from examples import afsim_twin_connector as connector
from opendis.PduFactory import createPdu
from opendis.dis7 import EntityStatePdu


@pytest.fixture
def args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["afsim_twin_connector.py"])
    return connector.parse_args()


def test_cli_defaults(args):
    assert vars(args) == {
        "destination": "127.0.0.1", "port": 3001, "entity_number": 1001,
        "site": 1, "application": 1, "marking": "TwinEntity",
        "latitude": 36.6, "longitude": -121.9, "altitude": 1000.0,
        "heading": 90.0, "speed_mps": 120.0, "turn_rate_dps": 0.5,
        "rate_hz": 2.0,
    }


def test_cli_overrides(monkeypatch):
    expected = {
        "destination": "192.0.2.10", "port": 4567, "entity_number": 42,
        "site": 3, "application": 4, "marking": "TestTwin",
        "latitude": -12.5, "longitude": 45.25, "altitude": 250.5,
        "heading": 270.5, "speed_mps": 15.5, "turn_rate_dps": -2.5,
        "rate_hz": 4.0,
    }
    argv = ["afsim_twin_connector.py"]
    for name, value in expected.items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    assert vars(connector.parse_args()) == expected


def build_sample(gps, **overrides):
    values = dict(
        entity_number=42, site=3, application=4, marking="ABCDEFGHIJKLM",
        latitude_deg=30.0, longitude_deg=-60.0, altitude_m=250.0,
        yaw_deg=60.0, speed_mps=20.0, timestamp=0x100000007,
    )
    values.update(overrides)
    return connector.build_entity_state(gps, **values)


def test_entity_state_construction():
    gps = Mock()
    gps.llarpy2ecef.return_value = (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    pdu = build_sample(gps)
    gps.llarpy2ecef.assert_called_once_with(
        math.radians(30), math.radians(-60), 250.0, 0.0, 0.0, math.radians(60)
    )
    assert isinstance(pdu, EntityStatePdu)
    assert pdu.entityID.entityNumber == 42
    assert pdu.entityID.simulationAddress.site == 3
    assert pdu.entityID.simulationAddress.application == 4
    assert pdu.marking.charactersString() == "ABCDEFGHIJK"
    assert (pdu.entityLocation.x, pdu.entityLocation.y, pdu.entityLocation.z) == (1, 2, 3)
    assert (pdu.entityOrientation.psi, pdu.entityOrientation.theta,
            pdu.entityOrientation.phi) == (0.1, 0.2, 0.3)
    assert (pdu.entityLinearVelocity.x, pdu.entityLinearVelocity.y,
            pdu.entityLinearVelocity.z) == pytest.approx((10, 10 * math.sqrt(3), 0))
    assert pdu.timestamp == 7


def test_serialization_round_trip():
    pdu = build_sample(connector.GPS(), marking="Twin")
    packet = connector.serialize_pdu(pdu)
    assert isinstance(packet, bytes)
    assert len(packet) == 144
    decoded = createPdu(packet)
    assert isinstance(decoded, EntityStatePdu)
    assert decoded.entityID.entityNumber == 42
    assert decoded.entityID.simulationAddress.site == 3
    assert decoded.entityID.simulationAddress.application == 4
    assert decoded.marking.charactersString() == "Twin"
    assert decoded.timestamp == 7
    for field, components in (
        ("entityLocation", ("x", "y", "z")),
        ("entityOrientation", ("psi", "theta", "phi")),
        ("entityLinearVelocity", ("x", "y", "z")),
    ):
        for component in components:
            assert getattr(getattr(decoded, field), component) == pytest.approx(
                getattr(getattr(pdu, field), component)
            )


@pytest.mark.parametrize("heading,turn_rate", [(0.0, 0.0), (90.0, 0.0), (359.0, 4.0)])
def test_simulation_motion_and_udp_send(args, monkeypatch, heading, turn_rate):
    args.heading = heading
    args.turn_rate_dps = turn_rate
    args.latitude = 60.0
    args.longitude = -20.0
    args.destination = "192.0.2.10"
    args.port = 4567
    args.speed_mps = 20.0
    args.rate_hz = 2.0
    sock = Mock()
    socket_factory = Mock(return_value=sock)
    monkeypatch.setattr(connector.socket, "socket", socket_factory)
    monkeypatch.setattr(connector.time, "time", Mock(side_effect=[3601.25, 3601.75, 3602.25]))

    class StopSimulation(Exception):
        pass

    sleep = Mock(side_effect=[None, None, StopSimulation])
    monkeypatch.setattr(connector.time, "sleep", sleep)
    with pytest.raises(StopSimulation):
        connector.run_simulation(args)

    socket_factory.assert_called_once_with(connector.socket.AF_INET, connector.socket.SOCK_DGRAM)
    assert sock.sendto.call_count == 3
    assert sleep.call_args_list == [((0.5,),)] * 3
    # Decode actual transmitted bytes: this exercises build/serialize/send together.
    latitude, longitude = 60.0, -20.0
    for index, call in enumerate(sock.sendto.call_args_list):
        packet, destination = call.args
        assert destination == ("192.0.2.10", 4567)
        assert isinstance(packet, bytes)
        pdu = createPdu(packet)
        assert isinstance(pdu, EntityStatePdu)
        assert pdu.timestamp == int((1.25 + index * 0.5) * 65536)
        expected = connector.GPS().llarpy2ecef(
            math.radians(latitude), math.radians(longitude), args.altitude,
            0.0, 0.0, math.radians(heading),
        )
        assert (pdu.entityLocation.x, pdu.entityLocation.y,
                pdu.entityLocation.z) == pytest.approx(expected[:3], abs=1e-6, rel=0)
        assert (pdu.entityLinearVelocity.x, pdu.entityLinearVelocity.y) == pytest.approx(
            (20 * math.cos(math.radians(heading)), 20 * math.sin(math.radians(heading))),
            abs=1e-5,
        )
        latitude += math.degrees(10 * math.cos(math.radians(heading)) / 6378137.0)
        longitude += math.degrees(
            10 * math.sin(math.radians(heading)) / (6378137.0 * math.cos(math.radians(latitude)))
        )
        heading = (heading + turn_rate * 0.5) % 360
