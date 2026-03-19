#!python
"""ACS (Autonomous Collaborative System) scenario: MQ-28 Ghost Bat + F-35A.

Simulates a 5-ship formation of 2x F-35A Lightning II flight leads with
3x MQ-28A Ghost Bat loyal wingmen operating as an autonomous collaborative
strike package.  Each platform is emitted as a DIS 7 EntityStatePdu at a
configurable rate so that an AFSIM DIS gateway can ingest the tracks in
real time.

Tactical datalink activity is represented by periodic TransmitterPdu and
SignalPdu messages (Link-16 style) between the F-35 leads and MQ-28 wingmen.

Usage examples
--------------
  # Default – loopback at 2 Hz, 5-ship formation heading east at FL300
  python acs_scenario_mq28_f35.py

  # Point at an AFSIM DIS gateway on the network
  python acs_scenario_mq28_f35.py --destination 192.168.1.50 --port 3000

  # Faster update rate, different starting position
  python acs_scenario_mq28_f35.py --rate-hz 5 --latitude -25.0 --longitude 134.0
"""

import argparse
import math
import socket
import time
from io import BytesIO
from typing import List, NamedTuple

from opendis.RangeCoordinates import GPS, deg2rad
from opendis.dis7 import EntityStatePdu, TransmitterPdu, SignalPdu
from opendis.stream import DataOutputStream

# ---------------------------------------------------------------------------
# DIS Entity-Type definitions (SISO-REF-010)
# ---------------------------------------------------------------------------
# EntityType fields: kind, domain, country, category, subcategory, specific, extra
#
# F-35A Lightning II  – US (225), Fighter/Air-Defense (1)
#   SISO enumeration 1.2.225.1.44.1.0
#
# MQ-28A Ghost Bat    – Australia (13), UAV (50 = Unmanned Aerial Vehicle)
#   No official SISO enum yet; representative values used.
# ---------------------------------------------------------------------------

FORCE_FRIENDLY = 1  # DIS ForceID – Friendly / Blue

# --- Platform descriptors ---------------------------------------------------

class PlatformSpec(NamedTuple):
    """Static description of one simulated platform."""
    entity_number: int
    marking: str        # up to 11 chars
    # EntityType fields
    kind: int
    domain: int
    country: int
    category: int
    subcategory: int
    specific: int
    extra: int
    # Formation offset from flight-lead reference point (NED metres)
    offset_north_m: float
    offset_east_m: float
    offset_alt_m: float
    # Role hint (for logging / AFSIM metadata)
    role: str


# Two F-35A flight leads
F35_LEAD = PlatformSpec(
    entity_number=101,
    marking="RAZOR11",
    kind=1, domain=2, country=225, category=1, subcategory=44, specific=1, extra=0,
    offset_north_m=0.0, offset_east_m=0.0, offset_alt_m=0.0,
    role="F-35A Lead",
)
F35_WING = PlatformSpec(
    entity_number=102,
    marking="RAZOR12",
    kind=1, domain=2, country=225, category=1, subcategory=44, specific=1, extra=0,
    offset_north_m=-500.0, offset_east_m=600.0, offset_alt_m=0.0,
    role="F-35A Wingman",
)

# Three MQ-28A Ghost Bat loyal wingmen
MQ28_1 = PlatformSpec(
    entity_number=201,
    marking="GHOST21",
    kind=1, domain=2, country=13, category=50, subcategory=1, specific=1, extra=0,
    offset_north_m=300.0, offset_east_m=-400.0, offset_alt_m=-150.0,
    role="MQ-28A #1",
)
MQ28_2 = PlatformSpec(
    entity_number=202,
    marking="GHOST22",
    kind=1, domain=2, country=13, category=50, subcategory=1, specific=1, extra=0,
    offset_north_m=300.0, offset_east_m=400.0, offset_alt_m=-150.0,
    role="MQ-28A #2",
)
MQ28_3 = PlatformSpec(
    entity_number=203,
    marking="GHOST23",
    kind=1, domain=2, country=13, category=50, subcategory=1, specific=1, extra=0,
    offset_north_m=600.0, offset_east_m=0.0, offset_alt_m=-300.0,
    role="MQ-28A #3",
)

ALL_PLATFORMS: List[PlatformSpec] = [F35_LEAD, F35_WING, MQ28_1, MQ28_2, MQ28_3]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6378137.0


def _serialize(pdu) -> bytes:
    buf = BytesIO()
    out = DataOutputStream(buf)
    pdu.serialize(out)
    return buf.getvalue()


def _apply_offset(lat_deg: float, lon_deg: float, alt_m: float,
                  heading_deg: float, spec: PlatformSpec):
    """Return (lat, lon, alt) after applying the formation offset rotated to heading."""
    heading_rad = math.radians(heading_deg)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    # Rotate offset from body-frame (north/east) to world-frame using heading
    world_north = spec.offset_north_m * cos_h - spec.offset_east_m * sin_h
    world_east = spec.offset_north_m * sin_h + spec.offset_east_m * cos_h

    new_lat = lat_deg + math.degrees(world_north / EARTH_RADIUS_M)
    new_lon = lon_deg + math.degrees(
        world_east / (EARTH_RADIUS_M * math.cos(math.radians(lat_deg)))
    )
    new_alt = alt_m + spec.offset_alt_m
    return new_lat, new_lon, new_alt


def build_entity_state(
    gps: GPS,
    spec: PlatformSpec,
    site: int,
    application: int,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    heading_deg: float,
    speed_mps: float,
    timestamp: int,
) -> EntityStatePdu:
    """Build an EntityStatePdu for one platform at its formation position."""
    p_lat, p_lon, p_alt = _apply_offset(lat_deg, lon_deg, alt_m, heading_deg, spec)

    pdu = EntityStatePdu()
    pdu.entityID.entityNumber = spec.entity_number
    pdu.entityID.simulationAddress.site = site
    pdu.entityID.simulationAddress.application = application
    pdu.forceId = FORCE_FRIENDLY

    # Entity type
    pdu.entityType.entityKind = spec.kind
    pdu.entityType.domain = spec.domain
    pdu.entityType.country = spec.country
    pdu.entityType.category = spec.category
    pdu.entityType.subcategory = spec.subcategory
    pdu.entityType.specific = spec.specific
    pdu.entityType.extra = spec.extra

    pdu.marking.setString(spec.marking[:11])

    loc = gps.llarpy2ecef(
        deg2rad(p_lat), deg2rad(p_lon), p_alt,
        0.0, 0.0, deg2rad(heading_deg),
    )
    pdu.entityLocation.x = loc[0]
    pdu.entityLocation.y = loc[1]
    pdu.entityLocation.z = loc[2]
    pdu.entityOrientation.psi = loc[3]
    pdu.entityOrientation.theta = loc[4]
    pdu.entityOrientation.phi = loc[5]

    heading_rad = math.radians(heading_deg)
    pdu.entityLinearVelocity.x = speed_mps * math.cos(heading_rad)
    pdu.entityLinearVelocity.y = speed_mps * math.sin(heading_rad)
    pdu.entityLinearVelocity.z = 0.0

    pdu.timestamp = timestamp & 0xFFFFFFFF
    return pdu


def build_transmitter_pdu(
    spec: PlatformSpec,
    site: int,
    application: int,
    timestamp: int,
) -> TransmitterPdu:
    """Build a Link-16 style TransmitterPdu for a platform."""
    pdu = TransmitterPdu()
    pdu.radioReferenceID.entityNumber = spec.entity_number
    pdu.radioReferenceID.simulationAddress.site = site
    pdu.radioReferenceID.simulationAddress.application = application
    pdu.radioNumber = 1

    # Link-16 operates in the 960-1215 MHz band (JTIDS/MIDS)
    pdu.frequency = 1065000000  # 1065 MHz representative centre frequency
    pdu.transmitFrequencyBandwidth = 3000000.0  # 3 MHz bandwidth
    pdu.power = 200.0  # dBm (representative)
    pdu.transmitState = 1  # Transmitting

    pdu.timestamp = timestamp & 0xFFFFFFFF
    return pdu


def build_signal_pdu(
    spec: PlatformSpec,
    site: int,
    application: int,
    timestamp: int,
    message_bytes: bytes,
) -> SignalPdu:
    """Build a SignalPdu carrying a short tactical data message."""
    pdu = SignalPdu()
    pdu.entityID.entityNumber = spec.entity_number
    pdu.entityID.simulationAddress.site = site
    pdu.entityID.simulationAddress.application = application
    pdu.radioID = 1

    # TDL type 1 = Link 16 Standard Message
    pdu.tdlType = 1
    pdu.encodingScheme = 2  # Raw binary / application-specific
    pdu.sampleRate = 0
    pdu.samples = len(message_bytes)
    pdu.data = list(message_bytes)

    pdu.timestamp = timestamp & 0xFFFFFFFF
    return pdu


# ---------------------------------------------------------------------------
# ACS behaviour phases
# ---------------------------------------------------------------------------

class ACSPhase:
    """Represents a named phase of the ACS mission."""
    def __init__(self, name: str, heading_deg: float, speed_mps: float,
                 turn_rate_dps: float, duration_s: float):
        self.name = name
        self.heading_deg = heading_deg
        self.speed_mps = speed_mps
        self.turn_rate_dps = turn_rate_dps
        self.duration_s = duration_s


DEFAULT_PHASES = [
    ACSPhase("TRANSIT",       heading_deg=90.0,  speed_mps=250.0, turn_rate_dps=0.0,  duration_s=120.0),
    ACSPhase("SPREAD",        heading_deg=90.0,  speed_mps=220.0, turn_rate_dps=0.3,  duration_s=60.0),
    ACSPhase("SEARCH",        heading_deg=45.0,  speed_mps=200.0, turn_rate_dps=0.5,  duration_s=180.0),
    ACSPhase("STRIKE_RUN_IN", heading_deg=0.0,   speed_mps=280.0, turn_rate_dps=0.0,  duration_s=90.0),
    ACSPhase("EGRESS",        heading_deg=180.0, speed_mps=260.0, turn_rate_dps=-0.2, duration_s=120.0),
]

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_scenario(args: argparse.Namespace) -> None:
    gps = GPS()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    dest = (args.destination, args.port)
    interval = 1.0 / args.rate_hz
    link16_interval = 2.0  # send datalink PDUs every 2 s

    latitude = args.latitude
    longitude = args.longitude
    heading_deg = args.heading
    speed_mps = args.speed

    # Build phase timeline
    if args.loop_phases:
        phases = DEFAULT_PHASES
    else:
        # Single continuous phase with CLI params
        phases = [ACSPhase("MANUAL", heading_deg, speed_mps, args.turn_rate_dps, float("inf"))]

    print("=" * 72)
    print("  ACS Scenario: MQ-28A Ghost Bat + F-35A Lightning II")
    print(f"  Target: {args.destination}:{args.port}  Rate: {args.rate_hz} Hz")
    print(f"  Platforms: {len(ALL_PLATFORMS)}")
    for p in ALL_PLATFORMS:
        print(f"    [{p.entity_number}] {p.marking:11s}  {p.role}")
    print("=" * 72)

    step = 0
    total_bytes = 0
    last_link16_time = 0.0

    for phase in phases:
        phase_start = time.time()
        heading_deg = phase.heading_deg
        speed_mps = phase.speed_mps

        print(f"\n>>> Phase: {phase.name}  hdg={phase.heading_deg}  "
              f"spd={phase.speed_mps} m/s  turn={phase.turn_rate_dps} deg/s  "
              f"dur={phase.duration_s}s")

        while True:
            now = time.time()
            elapsed_in_phase = now - phase_start
            if elapsed_in_phase >= phase.duration_s:
                break

            timestamp = int((now % 3600.0) * 65536)

            # --- Entity State PDUs for every platform -----------------------
            for spec in ALL_PLATFORMS:
                espdu = build_entity_state(
                    gps, spec, args.site, args.application,
                    latitude, longitude, args.altitude,
                    heading_deg, speed_mps, timestamp,
                )
                pkt = _serialize(espdu)
                sock.sendto(pkt, dest)
                total_bytes += len(pkt)

            # --- Link-16 datalink PDUs (throttled) --------------------------
            if now - last_link16_time >= link16_interval:
                last_link16_time = now
                # Each F-35 transmits; each MQ-28 transmits status back
                for spec in ALL_PLATFORMS:
                    tx = build_transmitter_pdu(spec, args.site, args.application, timestamp)
                    sock.sendto(_serialize(tx), dest)

                    # Simple tactical message: platform id + phase + position
                    msg = (f"{spec.marking}|{phase.name}|"
                           f"{latitude:.4f},{longitude:.4f}").encode("ascii")
                    sig = build_signal_pdu(spec, args.site, args.application, timestamp, msg)
                    sock.sendto(_serialize(sig), dest)

            # --- Advance formation reference point --------------------------
            distance = speed_mps * interval
            heading_rad = math.radians(heading_deg)
            d_north = distance * math.cos(heading_rad)
            d_east = distance * math.sin(heading_rad)

            latitude += math.degrees(d_north / EARTH_RADIUS_M)
            longitude += math.degrees(
                d_east / (EARTH_RADIUS_M * math.cos(math.radians(latitude)))
            )
            heading_deg = (heading_deg + phase.turn_rate_dps * interval) % 360.0

            if step % max(1, int(args.rate_hz * 5)) == 0:
                print(
                    f"  [{phase.name:14s}] step={step:06d}  lat={latitude:10.6f}  "
                    f"lon={longitude:11.6f}  hdg={heading_deg:6.1f}  "
                    f"total_KB={total_bytes / 1024:.1f}",
                    flush=True,
                )

            step += 1
            time.sleep(interval)

    print(f"\nScenario complete. {step} steps, {total_bytes / 1024:.1f} KB transmitted.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACS scenario injector: MQ-28A Ghost Bat + F-35A via DIS into AFSIM"
    )

    net = p.add_argument_group("Network")
    net.add_argument("--destination", default="127.0.0.1",
                     help="AFSIM DIS listener IP or multicast group (default: 127.0.0.1)")
    net.add_argument("--port", type=int, default=3001,
                     help="AFSIM DIS UDP port (default: 3001)")

    sim = p.add_argument_group("Simulation identity")
    sim.add_argument("--site", type=int, default=1, help="DIS site ID (default: 1)")
    sim.add_argument("--application", type=int, default=100,
                     help="DIS application ID (default: 100)")

    pos = p.add_argument_group("Initial conditions")
    pos.add_argument("--latitude", type=float, default=-23.8,
                     help="Formation centre latitude in degrees (default: -23.8, central Australia)")
    pos.add_argument("--longitude", type=float, default=133.9,
                     help="Formation centre longitude in degrees (default: 133.9)")
    pos.add_argument("--altitude", type=float, default=9144.0,
                     help="Altitude in metres (default: 9144 = FL300)")
    pos.add_argument("--heading", type=float, default=90.0,
                     help="Initial heading in degrees (default: 90 = east)")
    pos.add_argument("--speed", type=float, default=250.0,
                     help="Speed in m/s (default: 250, ~485 kts)")

    dyn = p.add_argument_group("Dynamics (manual mode)")
    dyn.add_argument("--turn-rate-dps", type=float, default=0.0,
                     help="Turn rate in deg/s (default: 0)")
    dyn.add_argument("--loop-phases", action="store_true",
                     help="Run pre-scripted mission phases instead of manual mode")

    rate = p.add_argument_group("Output")
    rate.add_argument("--rate-hz", type=float, default=2.0,
                      help="Entity-state PDU send rate in Hz (default: 2)")

    return p.parse_args()


if __name__ == "__main__":
    run_scenario(parse_args())
