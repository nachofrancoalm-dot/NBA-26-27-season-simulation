"""
serializers.py

Conversión de DataFrames de pandas a JSON serializable de verdad. pandas
representa "sin dato" con NaN/NaT, que Python's json.dumps puede escribir
como el literal `NaN` -- JSON válido para Python, pero JSON.parse() de
JavaScript lo rechaza. Toda respuesta de la API pasa por df_to_records()
antes de salir, para no filtrar ese detalle a cada router.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def df_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """DataFrame -> lista de dicts, con NaN/NaT convertidos a None. [] si df es None."""
    if df is None:
        return []
    clean = df.astype(object).where(pd.notnull(df), None)
    return clean.to_dict(orient="records")


def series_to_dict(series: Optional[pd.Series]) -> Dict[str, Any]:
    """Series (p.ej. value_counts()) -> dict {índice: valor}, NaN -> None. {} si series es None."""
    if series is None:
        return {}
    clean = series.where(pd.notnull(series), None)
    return {str(k): v for k, v in clean.to_dict().items()}
