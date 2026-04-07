from __future__ import annotations

from dataclasses import dataclass

from .models import ActionType, RobotAction


@dataclass
class BridgeEvent:
    message: str
    progress_subgoals: list[str]
    invalid_action: bool = False
    safety_violation: bool = False


class IsaacSimBridge:
    """
    Thin integration layer for Isaac Sim.

    In this hackathon scaffold, we provide a deterministic fallback simulator that
    mirrors expected robot-world transitions. If Isaac Sim is available, swap the
    methods in this class with omni.isaac calls while preserving output semantics.
    """

    VALID_LOCATIONS = {
        "start",
        "shelf_a",
        "shelf_b",
        "shelf_c",
        "staging_zone",
        "qc_station",
        "packing_zone",
        "fragile_pack_zone",
        "checkpoint_1",
        "dock",
    }

    def apply(
        self,
        action: RobotAction,
        *,
        task_id: str,
        robot_location: str,
        holding_object: str | None,
        object_locations: dict[str, str],
        battery_level: int,
    ) -> tuple[str, str | None, dict[str, str], int, BridgeEvent]:
        progress_subgoals: list[str] = []
        message = "Action processed."

        if action.action_type == ActionType.MOVE:
            if not action.target or action.target not in self.VALID_LOCATIONS:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("Invalid MOVE target.", [], invalid_action=True),
                )
            robot_location = action.target
            battery_level = max(0, battery_level - 8)
            message = f"Robot moved to {robot_location}."

        elif action.action_type == ActionType.PICK:
            if not action.target:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("PICK requires an object id target.", [], invalid_action=True),
                )
            obj = action.target
            obj_loc = object_locations.get(obj)
            if holding_object is not None:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("Gripper occupied.", [], invalid_action=True),
                )
            if obj_loc != robot_location:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("Object not reachable from current location.", [], invalid_action=True),
                )
            holding_object = obj
            object_locations[obj] = "gripper"
            battery_level = max(0, battery_level - 3)
            message = f"Picked {obj}."

        elif action.action_type == ActionType.PLACE:
            if holding_object is None:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("Cannot PLACE without object in gripper.", [], invalid_action=True),
                )
            if action.target and action.target != robot_location:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("PLACE target must match current location.", [], invalid_action=True),
                )
            object_locations[holding_object] = robot_location
            message = f"Placed {holding_object} at {robot_location}."
            holding_object = None
            battery_level = max(0, battery_level - 2)

        elif action.action_type == ActionType.SCAN:
            if not action.target:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("SCAN requires object id target.", [], invalid_action=True),
                )
            obj = action.target
            obj_loc = object_locations.get(obj)
            if obj_loc != robot_location and holding_object != obj:
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("Object not scannable from current location.", [], invalid_action=True),
                )
            battery_level = max(0, battery_level - 2)
            message = f"Scanned {obj}."

        elif action.action_type == ActionType.DOCK:
            if robot_location != "dock":
                return (
                    robot_location,
                    holding_object,
                    object_locations,
                    battery_level,
                    BridgeEvent("DOCK only valid at dock location.", [], invalid_action=True),
                )
            battery_level = 100
            message = "Docked and recharged."

        elif action.action_type == ActionType.WAIT:
            battery_level = max(0, battery_level - 1)
            message = "Waited one step."

        if battery_level == 0:
            return (
                robot_location,
                holding_object,
                object_locations,
                battery_level,
                BridgeEvent(
                    "Battery depleted; emergency stop.",
                    progress_subgoals,
                    safety_violation=True,
                ),
            )

        return (
            robot_location,
            holding_object,
            object_locations,
            battery_level,
            BridgeEvent(message=message, progress_subgoals=progress_subgoals),
        )
