#!python
"""Basic digital-twin connector that streams DIS EntityState PDUs.

This script generates a simple moving entity simulation and transmits it over UDP
so an AFSIM DIS listener can ingest the track.
"""

import argparse
import math
import socket
import time
from io import BytesIO

from opendis.RangeCoordinates import GPS, deg2rad
from opendis.dis7 import EntityStatePdu
from opendis.stream import DataOutputStream


def build_entity_state(
    gps: GPS,
    entity_number: int,
    site: int,
    application: int,
    marking: str,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    yaw_deg: float,
    speed_mps: float,
    timestamp: int,
) -> EntityStatePdu:
    """Create one EntityStatePdu sample for a moving twin entity."""
    pdu = EntityStatePdu()
    pdu.entityID.entityNumber = entity_number
    pdu.entityID.simulationAddress.site = site
    pdu.entityID.simulationAddress.application = application
    pdu.marking.setString(marking[:11])

    loc = gps.llarpy2ecef(
        deg2rad(latitude_deg),
        deg2rad(longitude_deg),
        altitude_m,
        0.0,
        0.0,
        deg2rad(yaw_deg),
    )

    pdu.entityLocation.x = loc[0]
    pdu.entityLocation.y = loc[1]
    pdu.entityLocation.z = loc[2]
    pdu.entityOrientation.psi = loc[3]
    pdu.entityOrientation.theta = loc[4]
    pdu.entityOrientation.phi = loc[5]

    pdu.entityLinearVelocity.x = speed_mps * math.cos(math.radians(yaw_deg))
    pdu.entityLinearVelocity.y = speed_mps * math.sin(math.radians(yaw_deg))
    pdu.entityLinearVelocity.z = 0.0
    pdu.timestamp = timestamp & 0xFFFFFFFF

    return pdu


def serialize_pdu(pdu: EntityStatePdu) -> bytes:
    stream = BytesIO()
    output = DataOutputStream(stream)
    pdu.serialize(output)
    return stream.getvalue()


def run_simulation(args: argparse.Namespace) -> None:
    gps = GPS()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    latitude = args.latitude
    longitude = args.longitude
    heading_deg = args.heading
    interval = 1.0 / args.rate_hz
    earth_radius = 6378137.0

    print(
        f"Starting twin connector -> {args.destination}:{args.port} "
        f"(entity={args.entity_number}, speed={args.speed_mps} m/s, rate={args.rate_hz} Hz)"
    )

    step = 0
    while True:
        now = time.time()
        timestamp = int((now % 3600.0) * 65536)

        pdu = build_entity_state(
            gps=gps,
            entity_number=args.entity_number,
            site=args.site,
            application=args.application,
            marking=args.marking,
            latitude_deg=latitude,
            longitude_deg=longitude,
            altitude_m=args.altitude,
            yaw_deg=heading_deg,
            speed_mps=args.speed_mps,
            timestamp=timestamp,
        )

        packet = serialize_pdu(pdu)
        sock.sendto(packet, (args.destination, args.port))

        print(
            f"step={step:05d} sent={len(packet)}B lat={latitude:.6f} lon={longitude:.6f} "
            f"heading={heading_deg:.1f}",
            flush=True,
        )

        distance = args.speed_mps * interval
        heading_rad = math.radians(heading_deg)
        d_north = distance * math.cos(heading_rad)
        d_east = distance * math.sin(heading_rad)

        latitude += math.degrees(d_north / earth_radius)
        longitude += math.degrees(d_east / (earth_radius * math.cos(math.radians(latitude))))
        heading_deg = (heading_deg + args.turn_rate_dps * interval) % 360.0

        step += 1
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic open-dis-python simulation for injecting tracks into AFSIM via DIS"
    )
    parser.add_argument("--destination", default="127.0.0.1", help="AFSIM listener IP or multicast group")
    parser.add_argument("--port", type=int, default=3001, help="AFSIM DIS UDP port")
    parser.add_argument("--entity-number", type=int, default=1001)
    parser.add_argument("--site", type=int, default=1)
    parser.add_argument("--application", type=int, default=1)
    parser.add_argument("--marking", default="TwinEntity")
    parser.add_argument("--latitude", type=float, default=36.6, help="Initial latitude in degrees")
    parser.add_argument("--longitude", type=float, default=-121.9, help="Initial longitude in degrees")
    parser.add_argument("--altitude", type=float, default=1000.0, help="Initial altitude in meters")
    parser.add_argument("--heading", type=float, default=90.0, help="Initial heading/yaw in degrees")
    parser.add_argument("--speed-mps", type=float, default=120.0, help="Speed in meters/sec")
    parser.add_argument("--turn-rate-dps", type=float, default=0.5, help="Turn rate in degrees/sec")
    parser.add_argument("--rate-hz", type=float, default=2.0, help="DIS send rate")
    return parser.parse_args()


if __name__ == "__main__":
    run_simulation(parse_args())
