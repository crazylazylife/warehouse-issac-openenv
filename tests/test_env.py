from warehouse_env.environment import WarehouseRobotEnv
from warehouse_env.models import RobotAction


def test_reset_and_state() -> None:
    env = WarehouseRobotEnv()
    obs = env.reset("easy_pick_and_stage")
    st = env.state()

    assert obs.task_id == "easy_pick_and_stage"
    assert st.task_id == "easy_pick_and_stage"
    assert st.step_count == 0


def test_easy_task_completion_script() -> None:
    env = WarehouseRobotEnv("easy_pick_and_stage")
    env.reset()

    env.step(RobotAction(action_type="move", target="shelf_a"))
    env.step(RobotAction(action_type="pick", target="tote_red"))
    env.step(RobotAction(action_type="move", target="staging_zone"))
    result = env.step(RobotAction(action_type="place", target="staging_zone"))

    assert result.done is True
    assert result.info["score"] >= 0.9
