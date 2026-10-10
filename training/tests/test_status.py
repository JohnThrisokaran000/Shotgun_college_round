"""The status table: sessions per group, and the note when two sessions post at once."""

import time
import unittest

from training import status


def group(starts, steps, now):
    return {"evals": [], "starts": starts, "steps": steps, "state": "training", "errors": 0, "gpus": 2, "seen": now}


class SessionTests(unittest.TestCase):
    def test_a_probe_and_its_training_start_are_one_session(self):
        self.assertEqual(len(status.sessions([100.0, 220.0])), 1)

    def test_starts_hours_apart_are_separate_sessions(self):
        self.assertEqual(len(status.sessions([100.0, 220.0, 4000.0, 4100.0])), 2)

    def test_two_interleaved_sessions_get_a_note(self):
        now = time.time()
        # the newest session started 10 minutes ago; an older one (steps 3000, 3070) still posts between its steps
        steps = [(now - 500, 60), (now - 480, 3000), (now - 440, 130), (now - 400, 3070), (now - 360, 200)]
        out = status.render({"acct01": group([now - 600], steps, now)}, now=now)
        self.assertIn("note: acct01: step counts go backwards", out)
        self.assertIn("two sessions are posting", out)

    def test_an_old_session_that_has_stopped_gets_no_note(self):
        now = time.time()
        # the old session posted only before the newest one started, which is what a finished first wave looks like
        steps = [(now - 3000, 1500), (now - 2900, 2000), (now - 500, 60), (now - 440, 130), (now - 380, 200)]
        out = status.render({"acct02": group([now - 3100, now - 600], steps, now)}, now=now)
        self.assertNotIn("note:", out)
        self.assertIn(time.strftime("%H:%M", time.localtime(now - 600)), out)

    def test_one_session_gets_no_note(self):
        now = time.time()
        steps = [(now - 200, 60), (now - 150, 130), (now - 100, 200)]
        out = status.render({"acct03": group([now - 300, now - 240], steps, now)}, now=now)
        self.assertNotIn("note:", out)


if __name__ == "__main__":
    unittest.main()
