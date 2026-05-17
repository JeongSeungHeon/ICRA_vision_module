import unittest
from types import SimpleNamespace

from object_pt_extraction.segmentation_engine import select_instances


def make_instance(class_name, score):
    return SimpleNamespace(class_name=class_name, score=float(score))


class SegmentationSelectionTests(unittest.TestCase):
    def test_prefers_wine_glass_over_higher_score_cup(self):
        instances = [
            make_instance("cup", 0.95),
            make_instance("wine glass", 0.40),
        ]

        selected = select_instances(
            instances,
            mode="highest_score",
            prefer_wine_glass_over_cup=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].class_name, "wine glass")

    def test_cup_only_keeps_existing_highest_score_behavior(self):
        instances = [
            make_instance("cup", 0.70),
            make_instance("cup", 0.90),
        ]

        selected = select_instances(
            instances,
            mode="highest_score",
            prefer_wine_glass_over_cup=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].score, 0.90)

    def test_wine_glass_only_keeps_existing_highest_score_behavior(self):
        instances = [
            make_instance("wine glass", 0.40),
            make_instance("wine glass", 0.80),
        ]

        selected = select_instances(
            instances,
            mode="highest_score",
            prefer_wine_glass_over_cup=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].score, 0.80)

    def test_cup_and_other_class_keep_existing_highest_score_behavior(self):
        instances = [
            make_instance("cup", 0.70),
            make_instance("box-shaped snack", 0.92),
        ]

        selected = select_instances(
            instances,
            mode="highest_score",
            prefer_wine_glass_over_cup=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].class_name, "box-shaped snack")


if __name__ == "__main__":
    unittest.main()
