"""CRI Benchmark — Contextual Resonance Index.

An open-source benchmark for evaluating AI long-term memory systems,
with a focus on ontology-based memory architectures.

The CRI Benchmark measures how effectively a system preserves meaningful
context across time, updates knowledge when new information appears,
resolves conflicting information, and maintains coherent representations
of entities and relationships.
"""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("cri-benchmark")
