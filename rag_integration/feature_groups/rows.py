"""Row-wise view of PythonDict input data."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.user.python_dict import columnar_to_rows, is_columnar


def as_rows(data: Any) -> List[Dict[str, Any]]:
    """Return ``data`` as row dicts; mloda hands PythonDict data over columnar."""
    if is_columnar(data):
        return columnar_to_rows(data)
    rows: List[Dict[str, Any]] = data
    return rows
