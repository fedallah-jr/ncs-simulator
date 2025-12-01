from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .packet import Packet


class ChannelState(Enum):
    IDLE = 0
    BUSY = 1


class EntityState(Enum):
    IDLE = 0
    BACKING_OFF = 1
    TRANSMITTING = 2


class NetworkEntity:
    """Represents a network entity (sensor or controller)."""

    def __init__(self, entity_id: int, entity_type: str):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.state = EntityState.IDLE
        self.pending_packet: Optional[Packet] = None
        self.backoff_counter = 0
        self.collision_count = 0


class NetworkModel:
    """
    Shared network implementing a simplified CSMA/CA-like contention scheme.
    Sensors and controllers compete for the medium; collisions drop packets.
    """

    def __init__(
        self,
        n_agents: int,
        data_rate_kbps: float = 250.0,
        data_packet_size: int = 50,
        ack_packet_size: int = 10,
        backoff_range: Tuple[int, int] = (0, 15),
        max_queue_size: int = 1,
        timestep_duration: float = 0.01,
        rng: np.random.Generator = None,
    ):
        self.n_agents = n_agents
        self.data_rate_kbps = data_rate_kbps
        self.data_packet_size = data_packet_size
        self.ack_packet_size = ack_packet_size
        self.backoff_range = backoff_range
        self.max_queue_size = max_queue_size
        self.timestep_duration = timestep_duration
        # Use provided RNG or create a new isolated one
        self.rng = rng if rng is not None else np.random.default_rng()

        self.data_tx_duration = self._compute_tx_duration(self.data_packet_size)
        self.ack_tx_duration = self._compute_tx_duration(self.ack_packet_size)

        self.channel_state = ChannelState.IDLE
        self.current_transmitter: Optional[int] = None
        self.transmission_end_timestep: Optional[int] = None
        self.current_timestep = 0
        self.total_collided_packets = 0

        self.entities: List[NetworkEntity] = []
        for i in range(n_agents):
            self.entities.append(NetworkEntity(entity_id=i, entity_type="sensor"))
        for i in range(n_agents):
            self.entities.append(NetworkEntity(entity_id=i, entity_type="controller"))

    def queue_data_packet(
        self, sensor_id: int, state_measurement: np.ndarray, measurement_timestamp: int
    ) -> Optional[Packet]:
        """Queue a data packet from a sensor.

        Returns:
            The overwritten packet if one was dropped, None otherwise.
        """
        entity_idx = sensor_id
        entity = self.entities[entity_idx]

        overwritten_packet = None
        if entity.pending_packet is not None and self.max_queue_size == 1:
            overwritten_packet = entity.pending_packet
            entity.pending_packet = None

        packet = Packet(
            source_id=sensor_id,
            dest_id=sensor_id,
            packet_type="data",
            payload={"state": state_measurement, "timestamp": measurement_timestamp},
            size_bytes=self.data_packet_size,
            timestamp_sent=self.current_timestep,
        )
        entity.pending_packet = packet

        # Always re-sense channel and set backoff when queuing a new packet
        # (Even if already backing off - new packet should get fresh carrier sense)
        if entity.state == EntityState.IDLE or entity.state == EntityState.BACKING_OFF:
            entity.state = EntityState.BACKING_OFF
            # True CSMA/CA: Sense channel before backoff
            if self.channel_state == ChannelState.IDLE:
                # Channel is free, transmit immediately (0) or after DIFS (1)
                entity.backoff_counter = 0
            else:
                # Channel is busy, use random backoff
                entity.backoff_counter = self._compute_backoff(entity.collision_count)

        return overwritten_packet

    def queue_ack_packet(self, controller_id: int, ack_data: Dict) -> Optional[Packet]:
        """Queue an ACK packet from a controller.

        Returns:
            The overwritten packet if one was dropped, None otherwise.
        """
        entity_idx = self.n_agents + controller_id
        entity = self.entities[entity_idx]

        overwritten_packet = None
        if entity.pending_packet is not None and self.max_queue_size == 1:
            overwritten_packet = entity.pending_packet
            entity.pending_packet = None

        packet = Packet(
            source_id=controller_id,
            dest_id=controller_id,
            packet_type="ack",
            payload=ack_data,
            size_bytes=self.ack_packet_size,
            timestamp_sent=self.current_timestep,
        )
        entity.pending_packet = packet

        # Always re-sense channel and set backoff when queuing a new packet
        # (Even if already backing off - new packet should get fresh carrier sense)
        if entity.state == EntityState.IDLE or entity.state == EntityState.BACKING_OFF:
            entity.state = EntityState.BACKING_OFF
            # True CSMA/CA: Sense channel before backoff
            if self.channel_state == ChannelState.IDLE:
                # Channel is free, transmit immediately (0) or after DIFS (1)
                entity.backoff_counter = 0
            else:
                # Channel is busy, use random backoff
                entity.backoff_counter = self._compute_backoff(entity.collision_count)

        return overwritten_packet

    def step(self) -> Dict[str, List[Packet]]:
        """
        Advance network simulation by one timestep (10 ms).

        Returns:
            Dict with delivered data packets, delivered ACK packets, and dropped packets.
        """
        self.current_timestep += 1
        delivered_data: List[Packet] = []
        delivered_acks: List[Packet] = []
        dropped_packets: List[Packet] = []

        for entity in self.entities:
            if entity.state == EntityState.BACKING_OFF and entity.backoff_counter > 0:
                entity.backoff_counter -= 1

        if self.channel_state == ChannelState.BUSY:
            if self.current_timestep >= (self.transmission_end_timestep or 0):
                transmitter = self.entities[self.current_transmitter]  # type: ignore[index]
                packet = transmitter.pending_packet

                if packet is not None:
                    if packet.packet_type == "data":
                        delivered_data.append(packet)
                    else:
                        delivered_acks.append(packet)

                transmitter.pending_packet = None
                transmitter.state = EntityState.IDLE
                transmitter.collision_count = 0
                self.channel_state = ChannelState.IDLE
                self.current_transmitter = None
                self.transmission_end_timestep = None

        ready_entities = []
        for idx, entity in enumerate(self.entities):
            if (
                entity.pending_packet is not None
                and entity.backoff_counter == 0
                and entity.state != EntityState.TRANSMITTING
            ):
                ready_entities.append(idx)

        if ready_entities:
            if self.channel_state == ChannelState.BUSY:
                for idx in ready_entities:
                    entity = self.entities[idx]
                    entity.state = EntityState.BACKING_OFF
                    entity.backoff_counter = self._compute_backoff(entity.collision_count)
            elif self.channel_state == ChannelState.IDLE:
                if len(ready_entities) == 1:
                    idx = ready_entities[0]
                    entity = self.entities[idx]
                    entity.state = EntityState.TRANSMITTING
                    self.channel_state = ChannelState.BUSY
                    self.current_transmitter = idx

                    duration = (
                        self.data_tx_duration
                        if entity.pending_packet and entity.pending_packet.packet_type == "data"
                        else self.ack_tx_duration
                    )
                    self.transmission_end_timestep = self.current_timestep + duration
                else:
                    # Collision: multiple entities tried to transmit simultaneously
                    for idx in ready_entities:
                        entity = self.entities[idx]
                        # Track the dropped packet before clearing it
                        if entity.pending_packet is not None:
                            dropped_packets.append(entity.pending_packet)
                        entity.pending_packet = None
                        entity.collision_count += 1
                        entity.state = EntityState.BACKING_OFF
                        entity.backoff_counter = self._compute_backoff(entity.collision_count)
                    self.total_collided_packets += len(ready_entities)

        return {
            "delivered_data": delivered_data,
            "delivered_acks": delivered_acks,
            "dropped_packets": dropped_packets,
        }

    def _compute_backoff(self, collision_count: int) -> int:
        """Compute a random backoff duration using exponential backoff."""
        min_backoff, max_backoff = self.backoff_range
        expanded_max = min(max_backoff * (2 ** collision_count), max_backoff * 16)
        return int(self.rng.integers(min_backoff, expanded_max + 1))

    def _compute_tx_duration(self, packet_size_bytes: int) -> int:
        """Compute number of timesteps needed to send a packet of given size."""
        bits = packet_size_bytes * 8
        bits_per_second = self.data_rate_kbps * 1000.0
        duration_seconds = bits / bits_per_second if bits_per_second > 0 else 0
        timestep = max(self.timestep_duration, 1e-9)
        timesteps = int(np.ceil(duration_seconds / timestep))
        return max(1, timesteps)

    def reset(self):
        """Reset network state."""
        self.channel_state = ChannelState.IDLE
        self.current_transmitter = None
        self.transmission_end_timestep = None
        self.current_timestep = 0
        self.total_collided_packets = 0

        for entity in self.entities:
            entity.state = EntityState.IDLE
            entity.pending_packet = None
            entity.backoff_counter = 0
            entity.collision_count = 0
