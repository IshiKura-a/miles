"""Small observable LRU cache used by resident GR services."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class CacheStats:
    capacity: int
    size: int
    hits: int
    misses: int
    evictions: int


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int = 8192) -> None:
        if capacity <= 0:
            raise ValueError("LRU cache capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[K, V] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> V | None:
        try:
            value = self._values.pop(key)
        except KeyError:
            self._misses += 1
            return None
        self._values[key] = value
        self._hits += 1
        return value

    def put(self, key: K, value: V) -> None:
        if key in self._values:
            self._values.pop(key)
        elif len(self._values) == self.capacity:
            self._values.popitem(last=False)
            self._evictions += 1
        self._values[key] = value

    def stats(self) -> CacheStats:
        return CacheStats(
            capacity=self.capacity,
            size=len(self._values),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )
