"""Batch sampler that keeps whole stories together.

Every within-setup ranking loss in the notebooks compares the two senses of
the same setup. With plain random batches of 32 drawn from ~2300 rows, a row's
sense partner lands in the same batch only ~1.3% of the time, so those losses
were almost never active. Batching by story puts all 6 rows (2 senses x 3
ending conditions) of each story in the same batch.
"""
import random
from collections import defaultdict


class StoryBatchSampler:
    """Yields lists of row indices; usable as DataLoader(batch_sampler=...) or directly.

    stories_per_batch=5 gives batches of 30 rows.
    """

    def __init__(self, story_ids, stories_per_batch=5, shuffle=True, seed=0):
        groups = defaultdict(list)
        for i, sid in enumerate(story_ids):
            groups[sid].append(i)
        self.groups = list(groups.values())
        self.stories_per_batch = stories_per_batch
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            self.rng.shuffle(groups)
        for start in range(0, len(groups), self.stories_per_batch):
            batch = [i for g in groups[start:start + self.stories_per_batch] for i in g]
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return (len(self.groups) + self.stories_per_batch - 1) // self.stories_per_batch


class RandomBatchSampler:
    """Plain shuffled batches (what the original notebooks did). Only for ablations."""

    def __init__(self, n, batch_size, seed=0):
        self.n, self.batch_size, self.rng = n, batch_size, random.Random(seed)

    def __iter__(self):
        order = list(range(self.n))
        self.rng.shuffle(order)
        for start in range(0, self.n, self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size
