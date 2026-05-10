import unittest

from perception.object_worker import TemporalClassLocker


class TemporalClassLockerTests(unittest.TestCase):
    def test_locks_after_stable_segmented_labels(self):
        locker = TemporalClassLocker(stable_frames=3)

        self.assertEqual(locker.update("cup", segmented=True), "cup")
        self.assertEqual(locker.update("cup", segmented=True), "cup")
        self.assertEqual(locker.update("cup", segmented=True), "cup")

        self.assertEqual(locker.locked_label, "cup")
        self.assertEqual(locker.update("wine glass", segmented=True), "cup")
        self.assertEqual(locker.update(None, segmented=False), "cup")

    def test_label_change_before_lock_delays_locking(self):
        locker = TemporalClassLocker(stable_frames=3)

        locker.update("cup", segmented=True)
        locker.update("wine glass", segmented=True)
        locker.update("wine glass", segmented=True)

        self.assertIsNone(locker.locked_label)
        self.assertEqual(locker.update("wine glass", segmented=True), "wine glass")
        self.assertEqual(locker.locked_label, "wine glass")

    def test_reset_clears_history_and_lock(self):
        locker = TemporalClassLocker(stable_frames=2)

        locker.update("box-shaped snack", segmented=True)
        locker.update("box-shaped snack", segmented=True)
        locker.reset()

        self.assertIsNone(locker.locked_label)
        self.assertEqual(tuple(locker.history), ())


if __name__ == "__main__":
    unittest.main()
