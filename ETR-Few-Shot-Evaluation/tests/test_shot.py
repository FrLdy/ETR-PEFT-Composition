import unittest

from icl.models import Shot, Shots


class TestShotsOrdering(unittest.TestCase):

    def setUp(self):
        self.shots = [
            Shot("A", 0.9, "translation"),
            Shot("B", 0.7, "summarization"),
            Shot("C", 0.8, "translation"),
            Shot("D", 0.6, "classification"),
            Shot("E", 0.5),
        ]
        self.task_order = ["summarization", "translation", "classification"]

    def test_sort_similarity_ascending(self):
        s = Shots(self.shots, self.task_order)
        ordered = s.order("sort_similarity_ascending")
        similarities = [shot.similariry for shot in ordered]
        self.assertEqual(similarities, sorted(similarities))

    def test_sort_similarity_descending(self):
        s = Shots(self.shots, self.task_order)
        ordered = s.order("sort_similarity_descending")
        similarities = [shot.similariry for shot in ordered]
        self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_alternating_tasks(self):
        s = Shots(self.shots, self.task_order)
        ordered = s.order("alternating_tasks")
        tasks = [shot.task for shot in ordered if shot.task is not None]
        # Test that consecutive tasks alternate as much as possible
        for i in range(1, len(tasks)):
            self.assertNotEqual(tasks[i], tasks[i - 1])

    def test_successive_tasks(self):
        s = Shots(self.shots, self.task_order)
        ordered = s.order("successive_tasks")
        # Check that tasks follow task_order
        current_order = [
            shot.task for shot in ordered if shot.task in self.task_order
        ]
        expected_order = []
        for task in self.task_order:
            expected_order.extend(
                [shot.task for shot in self.shots if shot.task == task]
            )
        self.assertEqual(current_order, expected_order)

    def test_successive_tasks_with_unknown_task(self):
        shots_with_unknown = self.shots + [Shot("F", 0.4, "extra_task")]
        s = Shots(shots_with_unknown, self.task_order)
        ordered = s.order("successive_tasks")
        known_tasks = [shot for shot in ordered if shot.task in self.task_order]
        extra_tasks = [shot for shot in ordered if shot.task == "extra_task"]
        # Check that extra_task appears after the known tasks
        self.assertGreater(
            ordered.shots.index(extra_tasks[0]),
            ordered.shots.index(known_tasks[-1]),
        )

    def test_shuffle(self):
        s = Shots(self.shots, self.task_order)
        ordered = s.order("shuffle")
        contents = [shot.content for shot in ordered]
        original_contents = [shot.content for shot in self.shots]
        self.assertCountEqual(
            contents, original_contents
        )  # Same elements, possibly different order
