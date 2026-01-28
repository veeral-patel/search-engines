"""Shared search utilities for Whoosh index queries."""

from __future__ import annotations

from typing import Dict, Iterable

from whoosh import scoring
from whoosh.qparser import MultifieldParser

DEFAULT_FIELD_BOOSTS: Dict[str, float] = {"title": 3.0, "authors": 2.0, "text": 1.0}
DEFAULT_FIELDS: Iterable[str] = ("title", "authors", "text")


def build_parser(ix, field_boosts: Dict[str, float] | None = None):
    """Build a query parser with optional field boosts."""
    boosts = field_boosts or DEFAULT_FIELD_BOOSTS
    return MultifieldParser(list(DEFAULT_FIELDS), schema=ix.schema, fieldboosts=boosts)


def build_searcher(ix, field_boosts: Dict[str, float] | None = None):
    """Build a searcher with BM25F scoring and optional field boosts."""
    boosts = field_boosts or DEFAULT_FIELD_BOOSTS
    return ix.searcher(weighting=scoring.BM25F(field_boosts=boosts))
