from dataclasses import dataclass
from typing import Any


@dataclass
class Packet:
    """Network packet representation."""

    source_id: int
    dest_id: int
    packet_type: str  # 'data', 'mac_ack', or 'app_ack'
    payload: Any
    size_bytes: int
    timestamp_sent: int
