import random
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Literal, Optional

ExamplesOrdering = Literal[
    "shuffle",
    "sort_similarity_ascending",
    "sort_similarity_descending",
    "alternating_tasks",
    "successive_tasks",
]


@dataclass
class Shot:
    content: Dict[str, Any]
    similariry: float
    task: Optional[str] = None


@dataclass
class Shots:
    shots: List[Shot]
    task_order: Optional[List[str]] = None

    def __iter__(self):
        return iter(self.shots)

    def order(self, method: ExamplesOrdering) -> "Shots":
        methods = {
            "shuffle": self._shuffle,
            "sort_similarity_ascending": self._sort_similarity_ascending,
            "sort_similarity_descending": self._sort_similarity_descending,
            "alternating_tasks": self._alternating_tasks,
            "successive_tasks": self._successive_tasks,
        }
        if method not in methods:
            raise ValueError(f"Unknown ordering method: {method}")
        shots = methods[method]()
        return Shots(shots=shots, task_order=self.task_order)

    def _shuffle(self) -> List[Shot]:
        shuffled = self.shots[:]
        random.shuffle(shuffled)
        return shuffled

    def _sort_similarity_ascending(self) -> List[Shot]:
        return sorted(self.shots, key=lambda shot: shot.similariry)

    def _sort_similarity_descending(self) -> List[Shot]:
        return sorted(
            self.shots, key=lambda shot: shot.similariry, reverse=True
        )

    def _alternating_tasks(self) -> List[Shot]:
        # Group shots by task
        task_groups = {}
        for shot in self.shots:
            task_groups.setdefault(shot.task, []).append(shot)

        # Remove shots with task=None
        task_groups = {k: v for k, v in task_groups.items() if k is not None}

        # Determine task cycling order
        if self.task_order:
            # Use only tasks that are present in task_groups
            task_keys = [
                task for task in self.task_order if task in task_groups
            ]
        else:
            task_keys = list(task_groups.keys())

        task_cycle = cycle(task_keys)
        ordered = []

        while any(task_groups.values()):
            for _ in range(len(task_keys)):  # Avoid infinite loops
                task = next(task_cycle)
                if task_groups[task]:
                    ordered.append(task_groups[task].pop(0))
                    break

        # Append shots with no task at the end
        none_tasks = [s for s in self.shots if s.task is None]
        ordered.extend(none_tasks)

        return ordered

    def _successive_tasks(self) -> List[Shot]:
        task_order = self.task_order
        if task_order is None:
            raise ValueError(
                "task_order must be provided for 'successive tasks' sorting."
            )

        # Group shots by task
        task_map = {task: [] for task in task_order}
        others = []

        for shot in self.shots:
            if shot.task in task_map:
                task_map[shot.task].append(shot)
            else:
                others.append(shot)

        # Flatten the ordered list
        ordered = []
        for task in task_order:
            ordered.extend(task_map[task])

        # Optionally, append the "unknown"/extra task shots at the end
        ordered.extend(others)
        return ordered

    def rename_tasks(self, rename_map: dict[str, str]) -> "Shots":
        # Crée une nouvelle liste de shots renommés
        new_shots = [
            Shot(
                content=shot.content,
                similariry=shot.similariry,
                task=rename_map.get(shot.task, shot.task),
            )
            for shot in self.shots
        ]

        # Met à jour l'ordre des tâches si nécessaire
        updated_order = self.task_order
        if self.task_order:
            updated_order = []
            seen = set()
            for task in self.task_order:
                new_task = rename_map.get(task, task)
                if new_task not in seen:
                    updated_order.append(new_task)
                    seen.add(new_task)

        return Shots(shots=new_shots, task_order=updated_order)
