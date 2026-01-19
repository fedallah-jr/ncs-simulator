from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


# Cached enum name lookups to avoid repeated .name property access in hot paths
_CHANNEL_STATE_NAMES: Dict[ChannelState, str] = {s: s.name for s in ChannelState}
_ENTITY_STATE_NAMES: Dict[EntityState, str] = {s: s.name for s in EntityState}


@dataclass
class ActiveTransmission:
    entity_idx: int
    packet: Packet
    end_slot: int
    collided: bool
    expects_ack: bool
    is_mac_ack: bool = False
    is_app_ack: bool = False
    ack_target_idx: Optional[int] = None
    acked_measurement_timestamp: Optional[int] = None


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
        self.ifs_countdown = 0


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
        mac_ifs_sifs_us: float = 192.0,
        mac_ifs_lifs_us: float = 640.0,
        mac_ifs_max_sifs_frame_size: int = 18,
        app_ack_enabled: bool = False,
        app_ack_packet_size: int = 30,
        app_ack_max_retries: int = 3,
        rng: np.random.Generator = None,
    ):
        self.n_agents = n_agents
        self.data_rate_kbps = data_rate_kbps
        self.data_packet_size = data_packet_size
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
        self.mac_ifs_max_sifs_frame_size = mac_ifs_max_sifs_frame_size
        self.mac_ifs_sifs_slots = (
            self._slots_from_seconds(mac_ifs_sifs_us * 1e-6) if mac_ifs_sifs_us > 0 else 0
        )
        self.mac_ifs_lifs_slots = (
            self._slots_from_seconds(mac_ifs_lifs_us * 1e-6) if mac_ifs_lifs_us > 0 else 0
        )

        self.rng = rng if rng is not None else np.random.default_rng()

        self.data_tx_slots = self._compute_tx_slots(self.data_packet_size)
        self.mac_ack_tx_slots = self._compute_tx_slots(self.mac_ack_size_bytes)

        self.app_ack_enabled = app_ack_enabled
        self.app_ack_packet_size = app_ack_packet_size
        self.app_ack_max_retries = app_ack_max_retries
        self.app_ack_tx_slots = self._compute_tx_slots(self.app_ack_packet_size)

        self.channel_state = ChannelState.IDLE
        self.current_transmitter: Optional[int] = None
        self.active_transmissions: List[ActiveTransmission] = []
        self.pending_mac_acks: List[Dict[str, int]] = []
        self.current_slot = 0
        self.delivered_mac_acks: List[Dict[str, int]] = []
        self.total_collided_packets = 0
        self.collisions_per_agent: List[int] = [0 for _ in range(n_agents)]
        self.data_delivered_total = 0
        self.data_delivered_per_agent: List[int] = [0 for _ in range(n_agents)]
        self.mac_ack_sent_total = 0
        self.mac_ack_sent_per_agent: List[int] = [0 for _ in range(n_agents)]
        self.mac_ack_collisions_total = 0
        self.mac_ack_collisions_per_agent: List[int] = [0 for _ in range(n_agents)]
        self.ack_timeouts_total = 0
        self.ack_timeouts_per_agent: List[int] = [0 for _ in range(n_agents)]
        self.app_ack_sent_total = 0
        self.app_ack_sent_per_agent: List[int] = [0] * n_agents
        self.app_ack_collisions_total = 0
        self.app_ack_collisions_per_agent: List[int] = [0] * n_agents
        self.app_ack_drops_total = 0
        self.app_ack_drops_per_agent: List[int] = [0] * n_agents
        self.app_ack_delivered_total = 0
        self.app_ack_delivered_per_agent: List[int] = [0] * n_agents
        self.delivered_app_acks: List[Dict[str, int]] = []
        self.trace_enabled = False
        self._trace_active = False
        self._trace_tick: Optional[int] = None
        self._trace_slot_index = 0
        self._trace_slots: List[Dict[str, Any]] = []
        self._trace_current_events: List[Dict[str, Any]] = []
        self._trace_counters: Dict[str, int] = {}

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

    def _get_ifs_slots(self, packet_size_bytes: int) -> int:
        if packet_size_bytes <= self.mac_ifs_max_sifs_frame_size:
            return self.mac_ifs_sifs_slots
        return self.mac_ifs_lifs_slots

    def _start_ifs(self, entity: NetworkEntity, packet_size_bytes: int) -> None:
        ifs_slots = self._get_ifs_slots(packet_size_bytes)
        if ifs_slots <= 0:
            return
        # Offset so the countdown cannot expire in the same slot it starts.
        entity.ifs_countdown = ifs_slots + 1

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
        if entity.pending_packet is not None:
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

        self._prepare_new_packet(entity_idx, entity, expects_ack=True)

        return overwritten_packet

    def run_slot(self) -> Dict[str, List[Packet]]:
        """Advance the network by one micro-slot."""
        delivered_data: List[Packet] = []
        dropped_packets: List[Packet] = []
        self.delivered_mac_acks = []
        self.delivered_app_acks = []

        self._complete_transmissions(delivered_data, dropped_packets)
        self._handle_ack_timeouts(dropped_packets)

        mac_ack_candidates = self._collect_mac_ack_transmissions()
        if mac_ack_candidates:
            self._start_transmissions(mac_ack_candidates, [])
        self._tick_backoffs()
        ready_entities = self._collect_ready_entities(dropped_packets)
        self._start_transmissions([], ready_entities)

        if self._trace_active:
            self._snapshot_trace_slot()

        self.current_slot += 1
        return {
            "delivered_data": delivered_data,
            "dropped_packets": dropped_packets,
            "delivered_mac_acks": self.delivered_mac_acks,
            "delivered_app_acks": self.delivered_app_acks,
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
        self.collisions_per_agent = [0 for _ in range(self.n_agents)]
        self.data_delivered_total = 0
        self.data_delivered_per_agent = [0 for _ in range(self.n_agents)]
        self.mac_ack_sent_total = 0
        self.mac_ack_sent_per_agent = [0 for _ in range(self.n_agents)]
        self.mac_ack_collisions_total = 0
        self.mac_ack_collisions_per_agent = [0 for _ in range(self.n_agents)]
        self.ack_timeouts_total = 0
        self.ack_timeouts_per_agent = [0 for _ in range(self.n_agents)]
        self.delivered_app_acks = []
        self.app_ack_sent_total = 0
        self.app_ack_sent_per_agent = [0] * self.n_agents
        self.app_ack_collisions_total = 0
        self.app_ack_collisions_per_agent = [0] * self.n_agents
        self.app_ack_drops_total = 0
        self.app_ack_drops_per_agent = [0] * self.n_agents
        self.app_ack_delivered_total = 0
        self.app_ack_delivered_per_agent = [0] * self.n_agents

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
            entity.ifs_countdown = 0
        self._trace_active = False
        self._trace_tick = None
        self._trace_slot_index = 0
        self._trace_slots = []
        self._trace_current_events = []
        self._trace_counters = {}

    def start_tick_trace(self, tick: int) -> None:
        if not self.trace_enabled:
            return
        self._trace_active = True
        self._trace_tick = int(tick)
        self._trace_slot_index = 0
        self._trace_slots = []
        self._trace_current_events = []
        self._trace_counters = {}

    def finish_tick_trace(self) -> Optional[Dict[str, Any]]:
        if not self._trace_active:
            return None
        trace = {
            "tick": int(self._trace_tick) if self._trace_tick is not None else None,
            "slots_per_step": int(self.slots_per_step),
            "slot_duration_s": float(self.slot_duration),
            "n_agents": int(self.n_agents),
            "entity_labels": self._trace_entity_labels(),
            "slots": list(self._trace_slots),
            "summary": dict(self._trace_counters),
        }
        self._trace_active = False
        self._trace_tick = None
        self._trace_slot_index = 0
        self._trace_slots = []
        self._trace_current_events = []
        self._trace_counters = {}
        return trace

    def _trace_entity_labels(self) -> List[str]:
        labels = [f"sensor_{i}" for i in range(self.n_agents)]
        labels.extend([f"controller_{i}" for i in range(self.n_agents)])
        return labels

    def _trace_event(self, event: Dict[str, Any]) -> None:
        if not self._trace_active:
            return
        self._trace_current_events.append(event)
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return
        if event_type == "collision":
            entities = event.get("entities")
            if isinstance(entities, list):
                self._trace_counters[event_type] = self._trace_counters.get(event_type, 0) + len(entities)
                return
        self._trace_counters[event_type] = self._trace_counters.get(event_type, 0) + 1

    def _snapshot_trace_slot(self) -> None:
        if not self._trace_active:
            return
        active_transmissions = [
            {
                "entity_idx": tx.entity_idx,
                "packet_type": tx.packet.packet_type,
                "collided": tx.collided,
                "end_slot": tx.end_slot,
                "is_mac_ack": tx.is_mac_ack,
                "is_app_ack": tx.is_app_ack,
            }
            for tx in self.active_transmissions
        ]
        slot_entry = {
            "slot": self._trace_slot_index,
            "global_slot": self.current_slot,
            "channel_state": _CHANNEL_STATE_NAMES[self.channel_state],
            "entity_states": [_ENTITY_STATE_NAMES[entity.state] for entity in self.entities],
            "active_transmissions": active_transmissions,
            "events": self._trace_current_events,
        }
        self._trace_slots.append(slot_entry)
        self._trace_current_events = []
        self._trace_slot_index += 1

    def _prepare_new_packet(self, entity_idx: int, entity: NetworkEntity, expects_ack: bool) -> None:
        """Initialize CSMA/CA bookkeeping for a freshly queued packet."""
        entity.state = EntityState.BACKING_OFF
        entity.backoff_exponent = self.mac_min_be
        entity.csma_backoffs = 0
        entity.retry_count = 0
        entity.collision_count = 0
        entity.awaiting_ack_until = None
        entity.cca_countdown = 0
        entity.expects_ack = expects_ack
        entity.backoff_counter = self._draw_backoff(entity.backoff_exponent, entity_idx=entity_idx)

    def _draw_backoff(self, backoff_exponent: int, *, entity_idx: Optional[int] = None) -> int:
        """
        Draw a random backoff for the given BE.

        We abide by the IEEE 802.15.4 standard where 1 Backoff Unit = 320 microseconds.
        We convert this physical duration into simulation slots.
        """
        upper = (2**backoff_exponent) - 1
        backoff_units = int(self.rng.integers(0, upper + 1))

        if backoff_units == 0:
            if self._trace_active and entity_idx is not None:
                self._trace_event(
                    {
                        "type": "backoff_draw",
                        "entity_idx": int(entity_idx),
                        "be": int(backoff_exponent),
                        "backoff_units": 0,
                        "backoff_slots": 0,
                    }
                )
            return 0

        # 320 us per unit
        duration_seconds = backoff_units * 320.0 * 1e-6

        # Convert to slots (ceiling to ensure we wait at least the duration)
        slots = int(np.ceil(duration_seconds / self.slot_duration))
        if self._trace_active and entity_idx is not None:
            self._trace_event(
                {
                    "type": "backoff_draw",
                    "entity_idx": int(entity_idx),
                    "be": int(backoff_exponent),
                    "backoff_units": int(backoff_units),
                    "backoff_slots": int(max(1, slots)),
                }
            )
        return max(1, slots)

    def _compute_tx_slots(self, packet_size_bytes: int) -> int:
        """Compute number of slots needed to send a packet of given size."""
        bits = packet_size_bytes * 8
        bits_per_second = self.data_rate_kbps * 1000.0
        duration_seconds = bits_per_second and bits / bits_per_second
        return max(1, self._slots_from_seconds(duration_seconds))

    def _tick_backoffs(self) -> None:
        for entity in self.entities:
            if entity.ifs_countdown > 0:
                entity.ifs_countdown -= 1
                continue
            if entity.state == EntityState.BACKING_OFF and entity.backoff_counter > 0:
                entity.backoff_counter -= 1

    def _collect_ready_entities(self, dropped_packets: List[Packet]) -> List[int]:
        """Move entities through CCA and return those cleared to transmit now."""
        ready: List[int] = []
        for idx, entity in enumerate(self.entities):
            if entity.pending_packet is None:
                continue

            if entity.ifs_countdown > 0:
                continue

            if entity.state == EntityState.BACKING_OFF and entity.backoff_counter <= 0:
                entity.state = EntityState.CCA
                entity.cca_countdown = self.cca_slots

            if entity.state == EntityState.CCA:
                if self.channel_state == ChannelState.BUSY and self.active_transmissions:
                    self._handle_cca_busy(idx, entity, dropped_packets)
                    continue

                entity.cca_countdown -= 1
                if entity.cca_countdown <= 0:
                    ready.append(idx)

        return ready

    def _handle_cca_busy(
        self, entity_idx: int, entity: NetworkEntity, dropped_packets: List[Packet]
    ) -> None:
        """Apply exponential backoff when CCA senses a busy channel."""
        entity.csma_backoffs += 1
        if self._trace_active:
            self._trace_event(
                {
                    "type": "cca_busy",
                    "entity_idx": int(entity_idx),
                    "csma_backoffs": int(entity.csma_backoffs),
                    "be": int(entity.backoff_exponent),
                }
            )
        if entity.csma_backoffs > self.max_csma_backoffs:
            if entity.pending_packet is not None:
                packet_type = entity.pending_packet.packet_type
                if packet_type == "app_ack":
                    sensor_id = entity.pending_packet.dest_id
                    self.app_ack_drops_total += 1
                    if 0 <= sensor_id < self.n_agents:
                        self.app_ack_drops_per_agent[sensor_id] += 1
                    if self._trace_active:
                        self._trace_event({"type": "app_ack_drop", "entity_idx": int(entity_idx), "reason": "csma_backoff"})
                else:
                    dropped_packets.append(entity.pending_packet)
                    if self._trace_active:
                        self._trace_event(
                            {
                                "type": "drop",
                                "entity_idx": int(entity_idx),
                                "reason": "csma_backoff",
                                "csma_backoffs": int(entity.csma_backoffs),
                            }
                        )
            self._clear_entity(entity)
            return

        entity.backoff_exponent = min(self.mac_max_be, entity.backoff_exponent + 1)
        entity.backoff_counter = self._draw_backoff(entity.backoff_exponent, entity_idx=entity_idx)
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

            is_app_ack = packet.packet_type == "app_ack"
            if is_app_ack:
                sensor_id = packet.dest_id
                self.app_ack_sent_total += 1
                if 0 <= sensor_id < self.n_agents:
                    self.app_ack_sent_per_agent[sensor_id] += 1

            new_transmissions.append(
                ActiveTransmission(
                    entity_idx=idx,
                    packet=packet,
                    end_slot=end_slot,
                    collided=False,
                    expects_ack=entity.expects_ack,
                    is_app_ack=is_app_ack,
                    ack_target_idx=packet.dest_id if is_app_ack else None,
                    acked_measurement_timestamp=packet.payload.get("acked_measurement_timestamp") if is_app_ack and packet.payload else None,
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
            collided_entities = [tx.entity_idx for tx in self.active_transmissions if tx.collided]
            collided_entities.extend(tx.entity_idx for tx in new_transmissions if tx.collided)
            if collided_entities and self._trace_active:
                self._trace_event({"type": "collision", "entities": collided_entities})

        collided_data = [
            tx for tx in new_transmissions
            if tx.collided and not tx.is_mac_ack and not tx.is_app_ack and tx.packet.packet_type == "data"
        ]
        self.total_collided_packets += len(collided_data)
        for tx in collided_data:
            sensor_id = int(tx.packet.source_id)
            if 0 <= sensor_id < self.n_agents:
                self.collisions_per_agent[sensor_id] += 1

        if self._trace_active:
            for tx in new_transmissions:
                self._trace_event(
                    {
                        "type": "tx_start",
                        "entity_idx": int(tx.entity_idx),
                        "packet_type": str(tx.packet.packet_type),
                        "is_mac_ack": bool(tx.is_mac_ack),
                        "is_app_ack": bool(tx.is_app_ack),
                        "end_slot": int(tx.end_slot),
                        "collided": bool(tx.collided),
                    }
                )

        self.active_transmissions.extend(new_transmissions)
        self.channel_state = ChannelState.BUSY

    def _get_tx_slots_for_packet(self, packet_type: str) -> int:
        if packet_type == "data":
            return self.data_tx_slots
        if packet_type == "app_ack":
            return self.app_ack_tx_slots
        return self.mac_ack_tx_slots

    def _complete_transmissions(
        self,
        delivered_data: List[Packet],
        dropped_packets: List[Packet],
    ) -> None:
        remaining: List[ActiveTransmission] = []
        for tx in self.active_transmissions:
            if tx.end_slot > self.current_slot:
                remaining.append(tx)
                continue

            entity = self.entities[tx.entity_idx]
            if tx.collided:
                if tx.is_mac_ack:
                    if self._trace_active:
                        self._trace_event(
                            {
                                "type": "mac_ack_collision",
                                "entity_idx": int(tx.entity_idx),
                                "ack_target_idx": int(tx.ack_target_idx)
                                if tx.ack_target_idx is not None
                                else None,
                            }
                        )
                    self.mac_ack_collisions_total += 1
                    if tx.ack_target_idx is not None:
                        ack_target_idx = int(tx.ack_target_idx)
                        if 0 <= ack_target_idx < self.n_agents:
                            self.mac_ack_collisions_per_agent[ack_target_idx] += 1
                    # MAC ACK collisions should not alter controller queue state.
                else:
                    if tx.is_app_ack:
                        sensor_id = tx.packet.dest_id
                        self.app_ack_collisions_total += 1
                        if 0 <= sensor_id < self.n_agents:
                            self.app_ack_collisions_per_agent[sensor_id] += 1
                    self._handle_failed_tx(tx.entity_idx, entity, dropped_packets)
            elif tx.is_mac_ack:
                if self._trace_active:
                    self._trace_event(
                        {
                            "type": "tx_complete",
                            "entity_idx": int(tx.entity_idx),
                            "packet_type": str(tx.packet.packet_type),
                            "is_mac_ack": True,
                        }
                    )
                if tx.ack_target_idx is not None:
                    self._mark_ack_received(tx.ack_target_idx)
            else:
                packet = tx.packet
                if packet.packet_type == "data":
                    delivered_data.append(packet)
                    dest_id = int(packet.dest_id)
                    if 0 <= dest_id < self.n_agents:
                        self.data_delivered_per_agent[dest_id] += 1
                        self.data_delivered_total += 1
                    receiver_idx = self.n_agents + packet.dest_id
                    measurement_timestamp = packet.payload.get("timestamp") if packet.payload else None

                    # MAC ACK always sent
                    self._schedule_mac_ack(receiver_idx, tx.entity_idx)

                    # App ACK also sent if enabled
                    if self.app_ack_enabled and measurement_timestamp is not None:
                        self._schedule_app_ack(receiver_idx, tx.entity_idx, measurement_timestamp)
                elif packet.packet_type == "app_ack":
                    # App ACK delivered successfully; sensor sends MAC ACK back.
                    sensor_id = int(packet.dest_id)
                    measurement_ts = tx.acked_measurement_timestamp
                    if measurement_ts is None and packet.payload:
                        measurement_ts = packet.payload.get("acked_measurement_timestamp")
                    self.delivered_app_acks.append(
                        {
                            "sensor_id": sensor_id,
                            "measurement_timestamp": measurement_ts,
                        }
                    )
                    if 0 <= sensor_id < self.n_agents:
                        self.app_ack_delivered_per_agent[sensor_id] += 1
                        self.app_ack_delivered_total += 1
                    self._schedule_mac_ack(sensor_id, tx.entity_idx)

                if self._trace_active:
                    trace_event = {
                        "type": "tx_complete",
                        "entity_idx": int(tx.entity_idx),
                        "packet_type": str(packet.packet_type),
                        "is_mac_ack": False,
                    }
                    if tx.is_app_ack:
                        trace_event["is_app_ack"] = True
                    self._trace_event(trace_event)

                if tx.expects_ack:
                    entity.state = EntityState.WAITING_ACK
                    entity.awaiting_ack_until = self.current_slot + self.mac_ack_wait_slots
                else:
                    ifs_packet_size = packet.size_bytes
                    self._clear_entity(entity)
                    self._start_ifs(entity, ifs_packet_size)

        self.active_transmissions = remaining
        self.channel_state = ChannelState.BUSY if self.active_transmissions else ChannelState.IDLE

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
            ack_target_idx = int(entry["ack_target_idx"])
            self.mac_ack_sent_total += 1
            if 0 <= ack_target_idx < self.n_agents:
                self.mac_ack_sent_per_agent[ack_target_idx] += 1
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

    def _schedule_app_ack(self, receiver_idx: int, tx_entity_idx: int, measurement_timestamp: int) -> None:
        """Queue app ACK - controller goes through full CSMA/CA."""
        if receiver_idx < 0 or receiver_idx >= len(self.entities):
            return

        controller_entity = self.entities[receiver_idx]

        # Don't queue if controller is already busy
        if controller_entity.pending_packet is not None:
            sensor_id = int(tx_entity_idx)
            self.app_ack_drops_total += 1
            if 0 <= sensor_id < self.n_agents:
                self.app_ack_drops_per_agent[sensor_id] += 1
            if self._trace_active:
                self._trace_event({"type": "app_ack_drop", "entity_idx": int(receiver_idx), "reason": "controller_busy"})
            return

        app_ack_packet = Packet(
            source_id=receiver_idx,
            dest_id=tx_entity_idx,  # sensor index
            packet_type="app_ack",
            payload={"acked_sensor_id": tx_entity_idx, "acked_measurement_timestamp": measurement_timestamp},
            size_bytes=self.app_ack_packet_size,
            timestamp_sent=self.current_slot,
        )

        controller_entity.pending_packet = app_ack_packet
        self._prepare_new_packet(receiver_idx, controller_entity, expects_ack=True)
        holdoff_slots = self.mac_ack_turnaround_slots + self.mac_ack_tx_slots
        if holdoff_slots > controller_entity.backoff_counter:
            controller_entity.backoff_counter = holdoff_slots

    def _handle_ack_timeouts(self, dropped_packets: List[Packet]) -> None:
        for entity_idx, entity in enumerate(self.entities):
            if entity.state == EntityState.WAITING_ACK and entity.awaiting_ack_until is not None:
                if self.current_slot >= entity.awaiting_ack_until:
                    if self._trace_active:
                        self._trace_event(
                            {
                                "type": "ack_timeout",
                                "entity_idx": int(entity_idx),
                            }
                        )
                    if 0 <= entity_idx < self.n_agents:
                        self.ack_timeouts_per_agent[entity_idx] += 1
                        self.ack_timeouts_total += 1
                    self._schedule_retry_or_drop(entity_idx, entity, dropped_packets, reason="ack_timeout")

    def _mark_ack_received(self, entity_idx: int) -> None:
        """Clear the transmitter once its MAC ACK is received."""
        entity = self.entities[entity_idx]
        ifs_packet_size: Optional[int] = None
        if entity.pending_packet is not None and entity.pending_packet.packet_type == "data":
            measurement_timestamp = entity.pending_packet.payload.get("timestamp")
            self.delivered_mac_acks.append(
                {
                    "sensor_id": entity.pending_packet.source_id,
                    "measurement_timestamp": measurement_timestamp,
                }
            )
        if entity.pending_packet is not None:
            ifs_packet_size = entity.pending_packet.size_bytes
        self._clear_entity(entity)
        if ifs_packet_size is not None:
            self._start_ifs(entity, ifs_packet_size)

    def _handle_failed_tx(
        self, entity_idx: int, entity: NetworkEntity, dropped_packets: List[Packet]
    ) -> None:
        """Handle a collided transmission by retrying or dropping."""
        entity.collision_count += 1
        self._schedule_retry_or_drop(entity_idx, entity, dropped_packets, reason="collision")

    def _schedule_retry_or_drop(
        self,
        entity_idx: int,
        entity: NetworkEntity,
        dropped_packets: List[Packet],
        *,
        reason: str,
    ) -> None:
        entity.retry_count += 1
        max_retries = self.max_frame_retries
        if entity.pending_packet is not None and entity.pending_packet.packet_type == "app_ack":
            max_retries = min(self.app_ack_max_retries, self.max_frame_retries)
        if entity.retry_count > max_retries:
            if entity.pending_packet is not None:
                if entity.pending_packet.packet_type == "app_ack":
                    sensor_id = entity.pending_packet.dest_id
                    self.app_ack_drops_total += 1
                    if 0 <= sensor_id < self.n_agents:
                        self.app_ack_drops_per_agent[sensor_id] += 1
                    if self._trace_active:
                        self._trace_event(
                            {
                                "type": "app_ack_drop",
                                "entity_idx": int(entity_idx),
                                "reason": str(reason),
                                "retry_count": int(entity.retry_count),
                            }
                        )
                else:
                    dropped_packets.append(entity.pending_packet)
                    if self._trace_active:
                        self._trace_event(
                            {
                                "type": "drop",
                                "entity_idx": int(entity_idx),
                                "reason": str(reason),
                                "retry_count": int(entity.retry_count),
                            }
                        )
            self._clear_entity(entity)
            return

        entity.state = EntityState.BACKING_OFF
        entity.backoff_exponent = self.mac_min_be
        entity.csma_backoffs = 0
        entity.backoff_counter = self._draw_backoff(entity.backoff_exponent, entity_idx=entity_idx)
        entity.cca_countdown = 0
        entity.awaiting_ack_until = None
        if self._trace_active:
            self._trace_event(
                {
                    "type": "retry",
                    "entity_idx": int(entity_idx),
                    "reason": str(reason),
                    "retry_count": int(entity.retry_count),
                    "be": int(entity.backoff_exponent),
                    "backoff_slots": int(entity.backoff_counter),
                }
            )

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
        entity.ifs_countdown = 0
