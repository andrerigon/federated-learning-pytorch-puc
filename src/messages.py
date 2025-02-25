from typing import TypedDict
import enum

class SimpleSender(enum.Enum):
    SENSOR = 0
    UAV = 1
    GROUND_STATION = 2

class SimpleMessage(TypedDict):
    packet_count: int
    sender_type: int
    sender: int
    payload: str
    type: str
    training_cycles: int
    model_updates: int
    success_rate: float
    global_model_version: int  # For messages from UAV to sensors
    local_model_version: int   # For messages from sensors to UAV
    extra_info: dict
