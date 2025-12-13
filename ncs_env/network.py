from __future__ import annotations

from dataclasses import dataclass
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
    CCA = 2
    TRANSMITTING = 3
    WAITING_ACK = 4


@dataclass
class ActiveTransmission:
    entity_idx: int
    packet: Packet
    end_slot: int
    collided: bool
    expects_ack: bool
    is_mac_ack: bool = False
    ack_target_idx: Optional[int] = None


class NetworkEntity:
    """Represents a network entity (sensor or controller)."""

    def __init__(self, entity_id: int, entity_type: str):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.state = EntityState.IDLE
        self.pending_packet: Optional[Packet] = None
        self.backoff_counter = 0
        self.collision_count = 0
        self.backoff_exponent = 3
        self.csma_backoffs = 0
        self.retry_count = 0
        self.cca_countdown = 0
        self.awaiting_ack_until: Optional[int] = None
        self.expects_ack = False


class NetworkModel:
    """
    Shared network implementing a micro-slot CSMA/CA contention scheme with MAC ACKs.
    Sensors and controllers compete for the medium; collisions trigger retries.
    """

    def __init__(
        self,
        n_agents: int,
        data_rate_kbps: float = 250.0,
        data_packet_size: int = 50,
        ack_packet_size: int = 10,
        max_queue_size: int = 1,
        timestep_duration: float = 0.01,
        slots_per_step: int = 32,
        mac_min_be: int = 3,
        mac_max_be: int = 5,
        max_csma_backoffs: int = 4,
        max_frame_retries: int = 3,
        mac_ack_wait_us: float = 864.0,
        mac_ack_turnaround_us: float = 192.0,
        cca_time_us: float = 128.0,
        mac_ack_size_bytes: int = 5,
        rng: np.random.Generator = None,
    ):
        self.n_agents = n_agents
        self.data_rate_kbps = data_rate_kbps
        self.data_packet_size = data_packet_size
        self.ack_packet_size = ack_packet_size
        self.max_queue_size = max_queue_size
        self.timestep_duration = timestep_duration
        self.slots_per_step = max(1, int(slots_per_step))
        self.slot_duration = self.timestep_duration / float(self.slots_per_step)

        self.mac_min_be = mac_min_be
        self.mac_max_be = mac_max_be
        self.max_csma_backoffs = max_csma_backoffs
        self.max_frame_retries = max_frame_retries
        self.mac_ack_size_bytes = mac_ack_size_bytes
        self.cca_slots = max(1, self._slots_from_seconds(cca_time_us * 1e-6))
        self.mac_ack_wait_slots = max(1, self._slots_from_seconds(mac_ack_wait_us * 1e-6))
        self.mac_ack_turnaround_slots = max(1, self._slots_from_seconds(mac_ack_turnaround_us * 1e-6))

        self.rng = rng if rng is not None else np.random.default_rng()

        self.data_tx_slots = self._compute_tx_slots(self.data_packet_size)
        self.ack_tx_slots = self._compute_tx_slots(self.ack_packet_size)
        self.mac_ack_tx_slots = self._compute_tx_slots(self.mac_ack_size_bytes)

        self.channel_state = ChannelState.IDLE
        self.current_transmitter: Optional[int] = None
        self.active_transmissions: List[ActiveTransmission] = []
        self.pending_mac_acks: List[Dict[str, int]] = []
        self.current_slot = 0
        self.delivered_mac_acks: List[Dict[str, int]] = []
        self.total_collided_packets = 0

        self.entities: List[NetworkEntity] = []
        for i in range(n_agents):
            self.entities.append(NetworkEntity(entity_id=i, entity_type="sensor"))
        for i in range(n_agents):
            self.entities.append(NetworkEntity(entity_id=i, entity_type="controller"))

    def _slots_from_seconds(self, duration_seconds: float) -> int:
        """Convert a duration in seconds to the corresponding number of slots."""
        if duration_seconds <= 0:
            return 1
        slots = int(np.ceil(duration_seconds / self.slot_duration))
        return max(1, slots)

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
            timestamp_sent=self.current_slot,
        )
        entity.pending_packet = packet

        self._prepare_new_packet(entity, expects_ack=True)

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
            timestamp_sent=self.current_slot,
        )
        entity.pending_packet = packet

        self._prepare_new_packet(entity, expects_ack=False)

        return overwritten_packet

    def run_slot(self) -> Dict[str, List[Packet]]:
        """Advance the network by one micro-slot."""
        delivered_data: List[Packet] = []
        delivered_acks: List[Packet] = []
        dropped_packets: List[Packet] = []
        self.delivered_mac_acks = []

        self._complete_transmissions(delivered_data, delivered_acks, dropped_packets)
        self._handle_ack_timeouts(dropped_packets)

        mac_ack_candidates = self._collect_mac_ack_transmissions()
        if mac_ack_candidates:
            self._start_transmissions(mac_ack_candidates, [])
        self._tick_backoffs()
        ready_entities = self._collect_ready_entities(dropped_packets)
        self._start_transmissions([], ready_entities)

        self.current_slot += 1
        return {
            "delivered_data": delivered_data,
            "delivered_acks": delivered_acks,
            "dropped_packets": dropped_packets,
            "delivered_mac_acks": self.delivered_mac_acks,
        }

    def reset(self):
        """Reset network state."""
        self.channel_state = ChannelState.IDLE
        self.current_transmitter = None
        self.active_transmissions = []
        self.pending_mac_acks = []
        self.current_slot = 0
        self.delivered_mac_acks = []
        self.total_collided_packets = 0

        for entity in self.entities:
            entity.state = EntityState.IDLE
            entity.pending_packet = None
            entity.backoff_counter = 0
            entity.collision_count = 0
            entity.backoff_exponent = self.mac_min_be
            entity.csma_backoffs = 0
            entity.retry_count = 0
            entity.awaiting_ack_until = None
            entity.cca_countdown = 0
            entity.expects_ack = False

    def _prepare_new_packet(self, entity: NetworkEntity, expects_ack: bool) -> None:
        """Initialize CSMA/CA bookkeeping for a freshly queued packet."""
        entity.state = EntityState.BACKING_OFF
        entity.backoff_exponent = self.mac_min_be
        entity.csma_backoffs = 0
        entity.retry_count = 0
        entity.collision_count = 0
        entity.awaiting_ack_until = None
        entity.cca_countdown = 0
        entity.expects_ack = expects_ack
        entity.backoff_counter = self._draw_backoff(entity.backoff_exponent)

    def _draw_backoff(self, backoff_exponent: int) -> int:
        """Draw a random backoff (number of slots) for the given BE."""
        upper = (2**backoff_exponent) - 1
        return int(self.rng.integers(0, upper + 1))

    def _compute_tx_slots(self, packet_size_bytes: int) -> int:
        """Compute number of slots needed to send a packet of given size."""
        bits = packet_size_bytes * 8
        bits_per_second = self.data_rate_kbps * 1000.0
        duration_seconds = bits_per_second and bits / bits_per_second
        return max(1, self._slots_from_seconds(duration_seconds))

    def _tick_backoffs(self) -> None:
        for entity in self.entities:
            if entity.state == EntityState.BACKING_OFF and entity.backoff_counter > 0:
                entity.backoff_counter -= 1

    def _collect_ready_entities(self, dropped_packets: List[Packet]) -> List[int]:
        """Move entities through CCA and return those cleared to transmit now."""
        ready: List[int] = []
        for idx, entity in enumerate(self.entities):
            if entity.pending_packet is None:
                continue

            if entity.state == EntityState.BACKING_OFF and entity.backoff_counter <= 0:
                entity.state = EntityState.CCA
                entity.cca_countdown = self.cca_slots

            if entity.state == EntityState.CCA:
                if self.channel_state == ChannelState.BUSY and self.active_transmissions:
                    self._handle_cca_busy(entity, dropped_packets)
                    continue

                entity.cca_countdown -= 1
                if entity.cca_countdown <= 0:
                    ready.append(idx)

        return ready

    def _handle_cca_busy(self, entity: NetworkEntity, dropped_packets: List[Packet]) -> None:
        """Apply exponential backoff when CCA senses a busy channel."""
        entity.csma_backoffs += 1
        if entity.csma_backoffs > self.max_csma_backoffs:
            if entity.pending_packet is not None:
                dropped_packets.append(entity.pending_packet)
            self._clear_entity(entity)
            return

        entity.backoff_exponent = min(self.mac_max_be, entity.backoff_exponent + 1)
        entity.backoff_counter = self._draw_backoff(entity.backoff_exponent)
        entity.state = EntityState.BACKING_OFF
        entity.cca_countdown = 0

    def _start_transmissions(
        self, mac_ack_candidates: List[ActiveTransmission], ready_entities: List[int]
    ) -> None:
        """Begin transmissions for entities cleared to send in this slot."""
        new_transmissions: List[ActiveTransmission] = list(mac_ack_candidates)

        for idx in ready_entities:
            entity = self.entities[idx]
            packet = entity.pending_packet
            if packet is None:
                continue

            tx_slots = self._get_tx_slots_for_packet(packet.packet_type)
            end_slot = self.current_slot + tx_slots
            new_transmissions.append(
                ActiveTransmission(
                    entity_idx=idx,
                    packet=packet,
                    end_slot=end_slot,
                    collided=False,
                    expects_ack=entity.expects_ack,
                )
            )
            entity.state = EntityState.TRANSMITTING

        if not new_transmissions:
            self.channel_state = ChannelState.BUSY if self.active_transmissions else ChannelState.IDLE
            return

        # Any overlapping transmissions collide
        if self.active_transmissions or len(new_transmissions) > 1:
            for tx in self.active_transmissions:
                tx.collided = True
            for tx in new_transmissions:
                tx.collided = True

        collided_count = sum(1 for tx in new_transmissions if tx.collided and not tx.is_mac_ack)
        self.total_collided_packets += collided_count

        self.active_transmissions.extend(new_transmissions)
        self.channel_state = ChannelState.BUSY

    def _get_tx_slots_for_packet(self, packet_type: str) -> int:
        if packet_type == "data":
            return self.data_tx_slots
        if packet_type == "ack":
            return self.ack_tx_slots
        return self.mac_ack_tx_slots

    def _complete_transmissions(
        self,
        delivered_data: List[Packet],
        delivered_acks: List[Packet],
        dropped_packets: List[Packet],
    ) -> None:
        remaining: List[ActiveTransmission] = []
        for tx in self.active_transmissions:
            if tx.end_slot > self.current_slot:
                remaining.append(tx)
                continue

            entity = self.entities[tx.entity_idx]
            if tx.collided:
                self._handle_failed_tx(entity, dropped_packets)
            elif tx.is_mac_ack:
                if tx.ack_target_idx is not None:
                    self._mark_ack_received(tx.ack_target_idx)
            else:
                packet = tx.packet
                if packet.packet_type == "data":
                    delivered_data.append(packet)
                    receiver_idx = self.n_agents + packet.dest_id
                    self._schedule_mac_ack(receiver_idx, tx.entity_idx)
                elif packet.packet_type == "ack":
                    delivered_acks.append(packet)

                if tx.expects_ack:
                    entity.state = EntityState.WAITING_ACK
                    entity.awaiting_ack_until = self.current_slot + self.mac_ack_wait_slots
                else:
                    self._clear_entity(entity)

        self.active_transmissions = remaining
        self.channel_state = ChannelState.BUSY if self.active_transmissions else ChannelState.IDLE

    def _mark_ack_received(self, entity_idx: int) -> None:
        """Clear the transmitter once its MAC ACK is received."""
        entity = self.entities[entity_idx]
        if entity.pending_packet is not None and entity.pending_packet.packet_type == "data":
            measurement_timestamp = entity.pending_packet.payload.get("timestamp")
            self.delivered_mac_acks.append(
                {
                    "sensor_id": entity.pending_packet.source_id,
                    "measurement_timestamp": measurement_timestamp,
                }
            )
        self._clear_entity(entity)

    def _handle_failed_tx(self, entity: NetworkEntity, dropped_packets: List[Packet]) -> None:
        """Handle a collided transmission by retrying or dropping."""
        entity.collision_count += 1
        self._schedule_retry_or_drop(entity, dropped_packets)

    def _schedule_retry_or_drop(self, entity: NetworkEntity, dropped_packets: List[Packet]) -> None:
        entity.retry_count += 1
        if entity.retry_count > self.max_frame_retries:
            if entity.pending_packet is not None:
                dropped_packets.append(entity.pending_packet)
            self._clear_entity(entity)
            return

        entity.state = EntityState.BACKING_OFF
        entity.backoff_exponent = min(self.mac_max_be, entity.backoff_exponent + 1)
        entity.backoff_counter = self._draw_backoff(entity.backoff_exponent)
        entity.cca_countdown = 0
        entity.awaiting_ack_until = None

    def _clear_entity(self, entity: NetworkEntity) -> None:
        entity.state = EntityState.IDLE
        entity.pending_packet = None
        entity.backoff_counter = 0
        entity.collision_count = 0
        entity.csma_backoffs = 0
        entity.retry_count = 0
        entity.backoff_exponent = self.mac_min_be
        entity.awaiting_ack_until = None
        entity.cca_countdown = 0
        entity.expects_ack = False

    def _collect_mac_ack_transmissions(self) -> List[ActiveTransmission]:
        ready: List[ActiveTransmission] = []
        remaining: List[Dict[str, int]] = []
        for entry in self.pending_mac_acks:
            if entry["start_slot"] > self.current_slot:
                remaining.append(entry)
                continue

            ack_packet = Packet(
                source_id=entry["source_idx"],
                dest_id=entry["ack_target_idx"],
                packet_type="mac_ack",
                payload=None,
                size_bytes=self.mac_ack_size_bytes,
                timestamp_sent=self.current_slot,
            )
            ready.append(
                ActiveTransmission(
                    entity_idx=entry["source_idx"],
                    packet=ack_packet,
                    end_slot=self.current_slot + self.mac_ack_tx_slots,
                    collided=False,
                    expects_ack=False,
                    is_mac_ack=True,
                    ack_target_idx=entry["ack_target_idx"],
                )
            )
        self.pending_mac_acks = remaining
        return ready

    def _schedule_mac_ack(self, receiver_idx: int, tx_entity_idx: int) -> None:
        if receiver_idx < 0 or receiver_idx >= len(self.entities):
            return
        start_slot = self.current_slot + self.mac_ack_turnaround_slots
        self.pending_mac_acks.append(
            {
                "source_idx": receiver_idx,
                "ack_target_idx": tx_entity_idx,
                "start_slot": start_slot,
            }
        )

    def _handle_ack_timeouts(self, dropped_packets: List[Packet]) -> None:
        for entity in self.entities:
            if entity.state == EntityState.WAITING_ACK and entity.awaiting_ack_until is not None:
                if self.current_slot >= entity.awaiting_ack_until:
                    self._schedule_retry_or_drop(entity, dropped_packets)
