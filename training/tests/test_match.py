import unittest

from training.match import summarize


class SummarizeTest(unittest.TestCase):
    def test_top_one_on_variants_with_a_target_and_refusal_on_the_rest(self):
        scores = [0.9, 0.2, 0.1, 0.8, 0.3, 0.2, 0.1]
        labels = [1, 0, 0, 0, 1, 0, 0]
        groups = ["a", "a", "a", "b", "b", "c", "c"]
        out = summarize(scores, labels, groups)
        # Group a and b have a target. The top scorer of a is the target, and of b it is not.
        self.assertEqual(out["variants_with_target"], 2)
        self.assertEqual(out["top1_present"], 0.5)
        # Group c has no target, and every score there is below 0.5, so it is refused.
        self.assertEqual(out["variants_without_target"], 1)
        self.assertEqual(out["refused_absent"], 1.0)

    def test_a_confident_wrong_score_on_an_absent_target_is_not_a_refusal(self):
        out = summarize([0.7], [0], ["x"])
        self.assertEqual(out["refused_absent"], 0.0)

    def test_empty_input_does_not_divide_by_zero(self):
        out = summarize([], [], [])
        self.assertEqual(out["top1_present"], 0.0)
        self.assertEqual(out["refused_absent"], 0.0)


if __name__ == "__main__":
    unittest.main()
