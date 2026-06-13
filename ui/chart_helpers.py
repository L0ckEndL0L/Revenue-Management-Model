from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def show_chart(chart_path: Path, caption: str) -> None:
    if chart_path.exists():
        st.image(str(chart_path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing chart: {caption}")


def to_datetime(df: pd.DataFrame, col: str = "stay_date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def interactive_line_chart(
    df: pd.DataFrame,
    y_columns: list[str],
    title: str,
    y_title: str,
    x_col: str = "stay_date",
) -> alt.Chart:
    chart_df = to_datetime(df, x_col)
    melted = chart_df.melt(id_vars=[x_col], value_vars=y_columns, var_name="series", value_name="value")
    return (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:T", title="Stay Date"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series:N", title="Series"),
            tooltip=[alt.Tooltip(f"{x_col}:T"), alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )


def interactive_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_title: str,
) -> alt.Chart:
    chart_df = to_datetime(df, x_col)
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:T", title="Stay Date"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[alt.Tooltip(f"{x_col}:T"), alt.Tooltip(f"{y_col}:Q", format=",.2f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )
