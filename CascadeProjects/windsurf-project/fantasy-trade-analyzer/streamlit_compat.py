import inspect
from typing import Any, Dict, Optional

import streamlit as st


_PLOTLY_SIG = None
_DATAFRAME_SIG = None


def plotly_chart(
    figure: Any,
    *,
    key: Optional[str] = None,
    width: str = "stretch",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    global _PLOTLY_SIG

    if _PLOTLY_SIG is None:
        try:
            _PLOTLY_SIG = inspect.signature(st.plotly_chart)
        except Exception:
            _PLOTLY_SIG = None

    params: Dict[str, Any] = {}

    if _PLOTLY_SIG is not None and "width" in _PLOTLY_SIG.parameters:
        params["width"] = width
    elif _PLOTLY_SIG is not None and "use_container_width" in _PLOTLY_SIG.parameters:
        params["use_container_width"] = width == "stretch"

    effective_config: Dict[str, Any] = {"responsive": True}
    if config:
        effective_config.update(config)

    if _PLOTLY_SIG is not None and "config" in _PLOTLY_SIG.parameters:
        params["config"] = effective_config

    if key is not None:
        params["key"] = key

    params.update(kwargs)
    return st.plotly_chart(figure, **params)


def dataframe(
    data: Any,
    *,
    width: str = "stretch",
    **kwargs: Any,
):
    global _DATAFRAME_SIG

    if _DATAFRAME_SIG is None:
        try:
            _DATAFRAME_SIG = inspect.signature(st.dataframe)
        except Exception:
            _DATAFRAME_SIG = None

    params: Dict[str, Any] = {}

    if _DATAFRAME_SIG is not None and "width" in _DATAFRAME_SIG.parameters:
        params["width"] = width
    elif _DATAFRAME_SIG is not None and "use_container_width" in _DATAFRAME_SIG.parameters:
        params["use_container_width"] = width == "stretch"

    params.update(kwargs)
    return st.dataframe(data, **params)
