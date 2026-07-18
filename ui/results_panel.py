from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluation import MONTH_ORDER
from src.yoy import summarize_yoy
from ui.chart_helpers import interactive_bar_chart, interactive_line_chart, show_chart
from ui.tailored_panel import format_optional_currency, format_optional_timestamp


def build_zip_bytes(output_dir: Path) -> bytes:
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(output_dir))
    memory_file.seek(0)
    return memory_file.read()


def safe_read_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_yoy_outputs(
    output_paths: Dict[str, str],
    pipeline_yoy_summary: Dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    existing_yoy = safe_read_csv(output_paths.get("yoy_comparison", ""))
    if existing_yoy is None or len(existing_yoy) == 0:
        return pd.DataFrame(), dict(pipeline_yoy_summary or {})

    return existing_yoy, dict(pipeline_yoy_summary or summarize_yoy(existing_yoy))


def format_yoy_change(value: float | None, prefix: str = "", suffix: str = "%") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{prefix}{value:+.1f}{suffix}"


def format_yoy_pair(current: float | None, prior: float | None, currency: bool = False, decimals: int = 1) -> str:
    if current is None or pd.isna(current) or prior is None or pd.isna(prior):
        return "No comparable prior-year data"

    if currency:
        return f"${current:,.{decimals}f} vs ${prior:,.{decimals}f}"
    return f"{current:.{decimals}f}% vs {prior:.{decimals}f}%"


def model_metric_value(metrics_df: pd.DataFrame, model: str, metric: str) -> float:
    if metrics_df is None or len(metrics_df) == 0 or metric not in metrics_df.columns:
        return float("nan")
    row = metrics_df[metrics_df.get("model", pd.Series(dtype=str)) == model]
    if len(row) == 0:
        return float("nan")
    return pd.to_numeric(row.iloc[0].get(metric), errors="coerce")


def sort_monthly_results(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return result rows in calendar-month order without changing columns."""
    if df is None or len(df) == 0 or "month" not in df.columns:
        return df
    month_rank = {month: rank for rank, month in enumerate(MONTH_ORDER)}
    out = df.copy()
    out["_month_rank"] = out["month"].map(month_rank).fillna(len(MONTH_ORDER))
    secondary = [col for col in ["day_type", "model"] if col in out.columns]
    return out.sort_values(["_month_rank", *secondary], kind="stable").drop(columns="_month_rank").reset_index(drop=True)


def filter_frame_to_month(df: pd.DataFrame | None, year: int, month: int) -> pd.DataFrame | None:
    """Filter dated output rows to the selected calendar month."""
    if df is None or len(df) == 0:
        return df
    date_col = next((col for col in ["stay_date", "calendar_date"] if col in df.columns), None)
    if date_col is None:
        if "month" in df.columns:
            month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B")
            return df[df["month"].astype(str) == month_name].reset_index(drop=True)
        return df
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return df[(dates.dt.year == year) & (dates.dt.month == month)].reset_index(drop=True)


def model_comparison_for_month(model_backtest_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if model_backtest_df is None or len(model_backtest_df) == 0:
        return None
    required = {"actual_rooms_sold", "baseline_rooms_sold", "enhanced_rooms_sold"}
    if not required.issubset(model_backtest_df.columns):
        return None
    actual = pd.to_numeric(model_backtest_df["actual_rooms_sold"], errors="coerce")
    rows = []
    for model, column in [("Baseline Model", "baseline_rooms_sold"), ("Tailored Model", "enhanced_rooms_sold")]:
        predicted = pd.to_numeric(model_backtest_df[column], errors="coerce")
        valid = actual.notna() & predicted.notna()
        errors = predicted[valid] - actual[valid]
        rows.append(
            {
                "model": model,
                "mae": float(errors.abs().mean()) if len(errors) else float("nan"),
                "rmse": float((errors.pow(2).mean()) ** 0.5) if len(errors) else float("nan"),
                "backtest_rows": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def render_results(
    output_paths: Dict[str, str],
    summary: Dict,
    use_interactive_charts: bool,
    timestamp: str,
) -> None:
    monthly_budget_df = safe_read_csv(output_paths.get("monthly_budget_summary", ""))
    if monthly_budget_df is None or len(monthly_budget_df) == 0:
        monthly_budget_df = pd.DataFrame(summary.get("monthly_budget_summaries", []))

    selected_year: int | None = None
    selected_month_number: int | None = None
    budget_summary = summary.get("budget_summary", {})
    if len(monthly_budget_df) > 0 and {"year", "month"}.issubset(monthly_budget_df.columns):
        monthly_budget_df = monthly_budget_df.copy()
        monthly_budget_df["year"] = pd.to_numeric(monthly_budget_df["year"], errors="coerce").astype("Int64")
        monthly_budget_df["month"] = pd.to_numeric(monthly_budget_df["month"], errors="coerce").astype("Int64")
        monthly_budget_df = monthly_budget_df.dropna(subset=["year", "month"]).sort_values(["year", "month"])
        month_choices = [
            (int(row.year), int(row.month), pd.Timestamp(year=int(row.year), month=int(row.month), day=1).strftime("%B %Y"))
            for row in monthly_budget_df.itertuples(index=False)
        ]
        selected_label = st.selectbox(
            "Summary month",
            options=[choice[2] for choice in month_choices],
            key="summary_month_selector",
        )
        selected_year, selected_month_number, _ = next(choice for choice in month_choices if choice[2] == selected_label)
        selected_budget = monthly_budget_df[
            (monthly_budget_df["year"] == selected_year) & (monthly_budget_df["month"] == selected_month_number)
        ]
        if len(selected_budget):
            budget_summary = selected_budget.iloc[0].to_dict()

    forecast_metrics = dict(summary.get("forecast_metrics", {}))
    projected_uplift = float(summary.get("projected_uplift_vs_baseline", 0.0))
    baseline_rows = int(summary.get("baseline_rows", 0))
    baseline_unavailable_rows = int(summary.get("baseline_unavailable_rows", 0))
    tailored_summary = dict(summary.get("tailored_summary", {}))

    if selected_year is not None and selected_month_number is not None:
        selected_forecast_actual = filter_frame_to_month(
            safe_read_csv(output_paths.get("forecast_vs_actual", "")), selected_year, selected_month_number
        )
        if selected_forecast_actual is not None and len(selected_forecast_actual) > 0:
            actual = pd.to_numeric(selected_forecast_actual.get("actual_rooms_sold"), errors="coerce")
            predicted = pd.to_numeric(selected_forecast_actual.get("enhanced_rooms_sold"), errors="coerce")
            valid = actual.notna() & predicted.notna()
            if valid.any():
                forecast_metrics["mae"] = float((predicted[valid] - actual[valid]).abs().mean())

        selected_rates = filter_frame_to_month(
            safe_read_csv(output_paths.get("rate_recommendations", "")), selected_year, selected_month_number
        )
        if selected_rates is not None and "uplift_vs_current" in selected_rates.columns:
            projected_uplift = float(pd.to_numeric(selected_rates["uplift_vs_current"], errors="coerce").fillna(0.0).sum())

        selected_baseline = filter_frame_to_month(
            safe_read_csv(output_paths.get("baseline_recommendations", "")), selected_year, selected_month_number
        )
        if selected_baseline is not None:
            baseline_rows = int(len(selected_baseline))
            baseline_unavailable_rows = int(
                (selected_baseline.get("baseline_status", pd.Series(dtype=str)) == "UNAVAILABLE").sum()
            )

        selected_tailored = filter_frame_to_month(
            safe_read_csv(output_paths.get("tailored_model_results", "")), selected_year, selected_month_number
        )
        if selected_tailored is not None and len(selected_tailored) > 0:
            source = selected_tailored.get("median_rate_source", pd.Series(dtype=str)).astype(str)
            tailored_summary.update(
                {
                    "avg_final_median_rate_used": pd.to_numeric(
                        selected_tailored.get("median_rate_used"), errors="coerce"
                    ).mean(),
                    "manual_daily_median_dates": int((source == "Manual daily input").sum()),
                    "dataset_derived_daily_median_dates": int((source == "Dataset-derived daily median").sum()),
                    "global_fallback_median_dates": int((source == "Global manual median fallback").sum()),
                    "missing_median_dates": int((source == "Missing median").sum()),
                }
            )

    st.subheader("Monthly Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly budget", f"${budget_summary.get('monthly_budget', 0.0):,.0f}")
    c2.metric("Revenue on books", f"${budget_summary.get('actual_revenue_to_date', 0.0):,.0f}")
    c3.metric("Month-end forecast", f"${budget_summary.get('month_end_forecast', 0.0):,.0f}")
    c4.metric("Budget variance", f"${budget_summary.get('variance_to_budget_abs', 0.0):,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Budget variance %", f"{budget_summary.get('variance_to_budget_pct', 0.0):+.1f}%")
    c6.metric("Required ADR remaining", f"${budget_summary.get('required_adr_remaining', 0.0):,.2f}")
    c7.metric("Forecast MAE", f"{forecast_metrics.get('mae', float('nan')):.2f}")
    c8.metric("Projected uplift", f"${projected_uplift:,.0f}")
    st.caption(
        "Baseline model rows: "
        f"{baseline_rows} "
        f"(unavailable: {baseline_unavailable_rows})"
    )
    if summary.get("using_uploaded_comparison", False):
        st.caption("YoY and baseline comparison are using the two uploaded datasets as the prior-year/current-year pair.")
    if budget_summary.get("budget_warning"):
        st.warning(f"Budget comparison skipped: {budget_summary['budget_warning']}")

    st.subheader("Tailored Model")
    t1, t2, t3 = st.columns(3)
    t1.metric("Avg final median used", format_optional_currency(tailored_summary.get("avg_final_median_rate_used")))
    t2.metric("Manual daily median dates", f"{int(tailored_summary.get('manual_daily_median_dates', 0))}")
    t3.metric("Global fallback dates", f"{int(tailored_summary.get('global_fallback_median_dates', 0))}")
    st.caption(
        "Median update frequency: "
        f"{tailored_summary.get('median_rate_update_frequency', 'Manual only')}"
        + " | Last update: "
        + format_optional_timestamp(tailored_summary.get("median_rate_last_updated"))
    )
    st.caption(
        "Daily median sources: "
        f"dataset-derived={int(tailored_summary.get('dataset_derived_daily_median_dates', 0))}, "
        f"missing={int(tailored_summary.get('missing_median_dates', 0))}"
    )

    yoy_df, yoy_summary = build_yoy_outputs(
        output_paths,
        summary.get("yoy_summary", {}),
    )
    if selected_year is not None and selected_month_number is not None and len(yoy_df) > 0:
        yoy_df = filter_frame_to_month(yoy_df, selected_year, selected_month_number)
        yoy_summary = summarize_yoy(yoy_df) if yoy_df is not None and len(yoy_df) > 0 else {}
    if len(yoy_df) > 0:
        st.subheader("Year-over-Year (YoY)")
        y1, y2, y3 = st.columns(3)
        y1.metric(
            "Occupancy Change",
            format_yoy_change(yoy_summary.get("occupancy_change_pct")),
            format_yoy_pair(
                yoy_summary.get("avg_current_occupancy_pct"),
                yoy_summary.get("avg_stly_occupancy_pct"),
            ),
        )
        y2.metric(
            "ADR Change",
            format_yoy_change(yoy_summary.get("adr_change_pct")),
            format_yoy_pair(
                yoy_summary.get("avg_current_adr"),
                yoy_summary.get("avg_stly_adr"),
                currency=True,
                decimals=2,
            ),
        )
        y3.metric(
            "Revenue Change",
            format_yoy_change(yoy_summary.get("revenue_change_pct")),
            format_yoy_pair(
                yoy_summary.get("total_current_revenue"),
                yoy_summary.get("total_stly_revenue"),
                currency=True,
                decimals=0,
            ),
        )
        st.caption(
            "YoY row status: "
            f"matched={yoy_summary.get('matched_rows', 0)}, "
            f"missing prior-year={yoy_summary.get('missing_rows', 0)}, "
            f"incomplete prior-year={yoy_summary.get('incomplete_rows', 0)}"
        )
        if not yoy_summary.get("has_comparable_data", False):
            st.info("No comparable prior-year stay dates were found for this run, so YoY change values are shown as N/A.")
    else:
        st.info("YoY data not available for this run. Add comparable STLY files under data/historical to enable YoY.")

    model_comparison_df = safe_read_csv(output_paths.get("baseline_vs_tailored_model_metrics", ""))
    if selected_year is not None and selected_month_number is not None:
        selected_model_backtest = safe_read_csv(output_paths.get("model_backtest_results", ""))
        selected_backtest_month_name = pd.Timestamp(
            year=selected_year, month=selected_month_number, day=1
        ).strftime("%B")
        if selected_model_backtest is not None and "month" in selected_model_backtest.columns:
            selected_model_backtest = selected_model_backtest[
                selected_model_backtest["month"].astype(str) == selected_backtest_month_name
            ].reset_index(drop=True)
        monthly_model_comparison = model_comparison_for_month(selected_model_backtest)
        if monthly_model_comparison is not None:
            model_comparison_df = monthly_model_comparison
    subgroup_metrics_df = sort_monthly_results(safe_read_csv(output_paths.get("subgroup_backtest_metrics", "")))
    selected_subgroup_metrics_df = subgroup_metrics_df
    rate_backtest_metrics_df = safe_read_csv(output_paths.get("rate_backtest_metrics", ""))
    rate_backtest_df = sort_monthly_results(safe_read_csv(output_paths.get("rate_backtest_results", "")))
    rate_subgroup_metrics_df = sort_monthly_results(safe_read_csv(output_paths.get("rate_subgroup_backtest_metrics", "")))
    selected_rate_backtest_df = rate_backtest_df
    selected_rate_subgroup_df = rate_subgroup_metrics_df
    intraday_changes_df = safe_read_csv(output_paths.get("intraday_recommendation_changes", ""))
    st.subheader("Baseline vs Tailored Model")
    if model_comparison_df is not None and len(model_comparison_df) > 0:
        b_mae = model_metric_value(model_comparison_df, "Baseline Model", "mae")
        t_mae = model_metric_value(model_comparison_df, "Tailored Model", "mae")
        b_rmse = model_metric_value(model_comparison_df, "Baseline Model", "rmse")
        t_rmse = model_metric_value(model_comparison_df, "Tailored Model", "rmse")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline MAE", f"{b_mae:.2f}")
        m2.metric("Tailored MAE", f"{t_mae:.2f}", f"{t_mae - b_mae:+.2f}")
        m3.metric("Baseline RMSE", f"{b_rmse:.2f}")
        m4.metric("Tailored RMSE", f"{t_rmse:.2f}", f"{t_rmse - b_rmse:+.2f}")
        st.dataframe(model_comparison_df, use_container_width=True)
        st.caption("Lower MAE and RMSE indicate better historical backtest accuracy.")
        warnings = model_comparison_df.get("validation_warning", pd.Series(dtype=str)).fillna("").astype(str)
        warnings = [warning for warning in warnings.unique().tolist() if warning]
        if warnings:
            st.warning(" ".join(warnings))
    else:
        st.info("Baseline vs tailored backtest metrics are not available for this run.")

    if subgroup_metrics_df is not None and len(subgroup_metrics_df) > 0:
        st.caption("Subgroup analysis by property type, event period, month, and weekday/weekend.")
        selected_month = (
            pd.Timestamp(year=selected_year, month=selected_month_number, day=1).strftime("%B")
            if selected_year is not None and selected_month_number is not None
            else "All months"
        )
        if selected_month != "All months":
            selected_subgroup_metrics_df = subgroup_metrics_df[
                subgroup_metrics_df["month"].astype(str) == selected_month
            ].reset_index(drop=True)
        st.dataframe(selected_subgroup_metrics_df, use_container_width=True)

    st.subheader("Rate Backtesting")
    if rate_backtest_metrics_df is not None and len(rate_backtest_metrics_df) > 0:
        st.dataframe(rate_backtest_metrics_df, use_container_width=True)
        st.caption(
            "Rolling backtest recommendations use only prior stay dates. Actual ADR is held out until scoring; "
            "negative differences mean lower error than baseline."
        )
        if rate_subgroup_metrics_df is not None and len(rate_subgroup_metrics_df) > 0:
            available_rate_months = [
                month for month in MONTH_ORDER if month in set(rate_subgroup_metrics_df["month"].astype(str))
            ]
            other_rate_months = sorted(
                set(rate_subgroup_metrics_df["month"].astype(str)) - set(available_rate_months)
            )
            rate_filter_1, rate_filter_2 = st.columns(2)
            selected_rate_month = (
                pd.Timestamp(year=selected_year, month=selected_month_number, day=1).strftime("%B")
                if selected_year is not None and selected_month_number is not None
                else "All months"
            )
            rate_filter_1.caption(f"Rate month: {selected_rate_month}")
            available_day_types = [
                day_type
                for day_type in ["Weekday", "Weekend"]
                if day_type in set(rate_subgroup_metrics_df["day_type"].astype(str))
            ]
            selected_rate_day_type = rate_filter_2.selectbox(
                "Rate day type",
                ["All day types", *available_day_types],
                key="rate_backtest_day_type_selector",
            )
            if selected_rate_month != "All months":
                selected_rate_subgroup_df = selected_rate_subgroup_df[
                    selected_rate_subgroup_df["month"].astype(str) == selected_rate_month
                ]
                if selected_rate_backtest_df is not None:
                    selected_rate_backtest_df = selected_rate_backtest_df[
                        selected_rate_backtest_df["month"].astype(str) == selected_rate_month
                    ]
            if selected_rate_day_type != "All day types":
                selected_rate_subgroup_df = selected_rate_subgroup_df[
                    selected_rate_subgroup_df["day_type"].astype(str) == selected_rate_day_type
                ]
                if selected_rate_backtest_df is not None:
                    selected_rate_backtest_df = selected_rate_backtest_df[
                        selected_rate_backtest_df["day_type"].astype(str) == selected_rate_day_type
                    ]
            selected_rate_subgroup_df = selected_rate_subgroup_df.reset_index(drop=True)
            if selected_rate_backtest_df is not None:
                selected_rate_backtest_df = selected_rate_backtest_df.reset_index(drop=True)
            st.dataframe(selected_rate_subgroup_df, use_container_width=True)
    else:
        st.info("Rate backtest metrics are not available for this run.")

    if intraday_changes_df is not None and len(intraday_changes_df) > 0:
        st.subheader("Intraday Recommendation Changes")
        st.dataframe(intraday_changes_df, use_container_width=True)
    elif summary.get("intraday_update_warnings"):
        st.warning("Intraday updates skipped: " + "; ".join(summary["intraday_update_warnings"]))

    st.subheader("Outputs")
    tabs = st.tabs([
        "Monthly Budget",
        "Forecast",
        "Baseline",
        "Tailored Model",
        "Rate Recommendations",
        "Top Raise",
        "Top Rescue",
        "Top Monitor",
        "Evaluation",
        "Baseline vs Tailored",
        "Subgroups",
        "Rate Backtest",
        "Intraday",
        "YoY",
    ])

    with tabs[0]:
        st.dataframe(monthly_budget_df if monthly_budget_df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[1]:
        df = safe_read_csv(output_paths.get("forecast", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[2]:
        df = safe_read_csv(output_paths.get("baseline_recommendations", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[3]:
        tailored_summary_df = safe_read_csv(output_paths.get("tailored_model_summary", ""))
        tailored_results_df = safe_read_csv(output_paths.get("tailored_model_results", ""))
        if tailored_summary_df is not None:
            st.dataframe(tailored_summary_df, use_container_width=True)
        st.dataframe(tailored_results_df if tailored_results_df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[4]:
        df = safe_read_csv(output_paths.get("rate_recommendations", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[5]:
        df = safe_read_csv(output_paths.get("top_raise_opportunities", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[6]:
        df = safe_read_csv(output_paths.get("top_rescue_dates", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[7]:
        df = safe_read_csv(output_paths.get("top_monitor_dates", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[8]:
        df = safe_read_csv(output_paths.get("evaluation_metrics", ""))
        st.dataframe(df if df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[9]:
        st.dataframe(model_comparison_df if model_comparison_df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[10]:
        st.caption("Use the month selector above to move between monthly results.")
        st.dataframe(
            selected_subgroup_metrics_df if selected_subgroup_metrics_df is not None else pd.DataFrame(),
            use_container_width=True,
        )
    with tabs[11]:
        st.caption("Use the rate month and day-type selectors above to navigate the rolling backtest.")
        if selected_rate_backtest_df is not None:
            st.dataframe(selected_rate_backtest_df, use_container_width=True)
        if selected_rate_subgroup_df is not None:
            st.dataframe(selected_rate_subgroup_df, use_container_width=True)
    with tabs[12]:
        st.dataframe(intraday_changes_df if intraday_changes_df is not None else pd.DataFrame(), use_container_width=True)
    with tabs[13]:
        st.dataframe(yoy_df if len(yoy_df) > 0 else pd.DataFrame(), use_container_width=True)

    st.subheader("Charts")
    run_dir = Path(output_paths["output_dir"])
    if use_interactive_charts:
        rate_df = safe_read_csv(output_paths.get("rate_recommendations", ""))
        priority_df = safe_read_csv(output_paths.get("top_monitor_dates", ""))
        forecast_vs_actual_df = safe_read_csv(output_paths.get("forecast_vs_actual", ""))
        model_comparison_df = safe_read_csv(output_paths.get("baseline_vs_tailored_model_metrics", ""))
        subgroup_metrics_df = selected_subgroup_metrics_df

        ch1, ch2 = st.columns(2)
        with ch1:
            if model_comparison_df is not None and {"model", "mae", "rmse"}.issubset(model_comparison_df.columns):
                metric_chart_df = model_comparison_df.melt(
                    id_vars=["model"],
                    value_vars=["mae", "rmse"],
                    var_name="metric",
                    value_name="value",
                )
                metric_chart = (
                    alt.Chart(metric_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("model:N", title="Model"),
                        y=alt.Y("value:Q", title="Error (Rooms Sold)"),
                        color=alt.Color("metric:N", title="Metric"),
                        xOffset="metric:N",
                        tooltip=[
                            alt.Tooltip("model:N", title="Model"),
                            alt.Tooltip("metric:N", title="Metric"),
                            alt.Tooltip("value:Q", title="Value", format=",.2f"),
                        ],
                    )
                    .properties(title="Baseline vs Tailored Model Accuracy", height=320)
                )
                st.altair_chart(metric_chart, use_container_width=True)
            else:
                show_chart(run_dir / "baseline_vs_tailored_model_metrics.png", "Baseline vs Tailored Model Accuracy")

            if rate_df is not None and {"stay_date", "current_rate", "recommended_rate"}.issubset(rate_df.columns):
                st.altair_chart(
                    interactive_line_chart(
                        rate_df,
                        y_columns=["current_rate", "recommended_rate"],
                        title="Current vs Recommended Rate",
                        y_title="Rate",
                    ),
                    use_container_width=True,
                )
            else:
                show_chart(run_dir / "current_vs_recommended_rate.png", "Current vs Recommended Rate")

            if priority_df is not None and {"stay_date", "priority_score"}.issubset(priority_df.columns):
                st.altair_chart(
                    interactive_line_chart(
                        priority_df,
                        y_columns=["priority_score"],
                        title="Priority Score by Date",
                        y_title="Priority Score",
                    ),
                    use_container_width=True,
                )
            else:
                show_chart(run_dir / "priority_score_by_date.png", "Priority Score by Date")

            if len(yoy_df) > 0 and {"calendar_date", "Current OCC %", "STLY OCC %"}.issubset(yoy_df.columns):
                yoy_chart_df = yoy_df[["calendar_date", "Current OCC %", "STLY OCC %"]].copy()
                yoy_melted = yoy_chart_df.melt(
                    id_vars=["calendar_date"],
                    value_vars=["Current OCC %", "STLY OCC %"],
                    var_name="series",
                    value_name="value",
                )
                yoy_chart = (
                    alt.Chart(yoy_melted)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("calendar_date:N", title="Calendar Date (MM-DD)"),
                        y=alt.Y("value:Q", title="Occupancy %"),
                        color=alt.Color("series:N", title="Series"),
                        tooltip=[
                            alt.Tooltip("calendar_date:N", title="Calendar Date"),
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("value:Q", title="Occupancy %", format=",.2f"),
                        ],
                    )
                    .properties(title="YoY Occupancy Comparison", height=320)
                    .interactive()
                )
                st.altair_chart(yoy_chart, use_container_width=True)

        with ch2:
            if subgroup_metrics_df is not None and {"property_type", "event_period", "model", "rmse"}.issubset(subgroup_metrics_df.columns):
                subgroup_chart_df = subgroup_metrics_df.copy()
                for col in ["month", "day_type"]:
                    if col not in subgroup_chart_df.columns:
                        subgroup_chart_df[col] = "Unspecified"
                subgroup_chart_df["subgroup"] = (
                    subgroup_chart_df["property_type"].astype(str)
                    + " | "
                    + subgroup_chart_df["event_period"].astype(str)
                    + " | "
                    + subgroup_chart_df["month"].astype(str)
                    + " | "
                    + subgroup_chart_df["day_type"].astype(str)
                )
                subgroup_chart = (
                    alt.Chart(subgroup_chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("subgroup:N", title="Property Type | Event Period | Month | Day Type", sort=None),
                        y=alt.Y("rmse:Q", title="RMSE (Rooms Sold)"),
                        color=alt.Color("model:N", title="Model"),
                        xOffset="model:N",
                        tooltip=[
                            alt.Tooltip("property_type:N", title="Property Type"),
                            alt.Tooltip("event_period:N", title="Event Period"),
                            alt.Tooltip("month:N", title="Month"),
                            alt.Tooltip("day_type:N", title="Day Type"),
                            alt.Tooltip("model:N", title="Model"),
                            alt.Tooltip("rmse:Q", title="RMSE", format=",.2f"),
                            alt.Tooltip("mae:Q", title="MAE", format=",.2f"),
                        ],
                    )
                    .properties(title="Subgroup Backtest RMSE", height=320)
                )
                st.altair_chart(subgroup_chart, use_container_width=True)
            else:
                show_chart(run_dir / "subgroup_backtest_metrics.png", "Subgroup Backtest RMSE")

            if rate_df is not None and {"stay_date", "uplift_vs_current"}.issubset(rate_df.columns):
                st.altair_chart(
                    interactive_bar_chart(
                        rate_df,
                        x_col="stay_date",
                        y_col="uplift_vs_current",
                        title="Expected Revenue Uplift",
                        y_title="Uplift",
                    ),
                    use_container_width=True,
                )
            else:
                show_chart(run_dir / "expected_revenue_uplift.png", "Expected Revenue Uplift")

            if forecast_vs_actual_df is not None and {
                "stay_date",
                "actual_rooms_sold",
                "baseline_rooms_sold",
                "enhanced_rooms_sold",
            }.issubset(forecast_vs_actual_df.columns):
                st.altair_chart(
                    interactive_line_chart(
                        forecast_vs_actual_df,
                        y_columns=["actual_rooms_sold", "baseline_rooms_sold", "enhanced_rooms_sold"],
                        title="Forecast vs Actual (Rooms Sold)",
                        y_title="Rooms Sold",
                    ),
                    use_container_width=True,
                )
            else:
                show_chart(run_dir / "forecast_vs_actual.png", "Forecast vs Actual")
    else:
        ch1, ch2 = st.columns(2)
        with ch1:
            show_chart(run_dir / "baseline_vs_tailored_model_metrics.png", "Baseline vs Tailored Model Accuracy")
            show_chart(run_dir / "current_vs_recommended_rate.png", "Current vs Recommended Rate")
            show_chart(run_dir / "priority_score_by_date.png", "Priority Score by Date")
            if len(yoy_df) > 0:
                st.info("YoY Occupancy Comparison is available when interactive charts are enabled.")
        with ch2:
            show_chart(run_dir / "subgroup_backtest_metrics.png", "Subgroup Backtest RMSE")
            show_chart(run_dir / "expected_revenue_uplift.png", "Expected Revenue Uplift")
            show_chart(run_dir / "forecast_vs_actual.png", "Forecast vs Actual")

    st.subheader("Downloads")
    for key in [
        "monthly_budget_summary",
        "forecast",
        "baseline_recommendations",
        "tailored_model_results",
        "tailored_model_summary",
        "rate_recommendations",
        "top_raise_opportunities",
        "top_rescue_dates",
        "top_monitor_dates",
        "evaluation_metrics",
        "baseline_vs_tailored_model_metrics",
        "subgroup_backtest_metrics",
        "rate_backtest_results",
        "rate_backtest_metrics",
        "rate_subgroup_backtest_metrics",
        "intraday_recommendation_changes",
        "baseline_vs_new_policy",
        "yoy_comparison",
    ]:
        file_path = output_paths.get(key)
        if file_path and Path(file_path).exists():
            st.download_button(
                f"Download {Path(file_path).name}",
                data=Path(file_path).read_bytes(),
                file_name=Path(file_path).name,
                mime="text/csv",
                key=f"download_{key}",
            )

    zip_bytes = build_zip_bytes(Path(output_paths["output_dir"]))
    st.download_button(
        "Download all outputs (zip)",
        data=zip_bytes,
        file_name=f"rateanchor_outputs_{timestamp}.zip",
        mime="application/zip",
    )
