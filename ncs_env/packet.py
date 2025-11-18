from dataclasses import dataclass
from typing import Any


@dataclass
class Packet:
    """Network packet representation."""

    source_id: int
    dest_id: int
    packet_type: str  # 'data' or 'ack'
    payload: Any
    size_bytes: int
    timestamp_sent: int
