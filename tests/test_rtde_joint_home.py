import unittest
import sys
import types

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *_args, **_kwargs: {}
sys.modules.setdefault("yaml", yaml_stub)
sys.modules.pop("robot.rtde_controller", None)
from robot.rtde_controller import RtdeController


class RtdeJointHomeTests(unittest.TestCase):
    def test_mock_joint_move_updates_joint_positions(self):
        controller = RtdeController(
            {
                "robot": {
                    "rtde": {
                        "force_mock": True,
                    },
                },
            }
        )
        target = (0.0, -2.35619449, 2.35619449, 0.0, 1.57079633, 0.0)

        self.assertTrue(controller.move_to_joint_positions(target))
        state = controller.read_robot_state()

        self.assertEqual(state.joint_positions, target)


if __name__ == "__main__":
    unittest.main()
