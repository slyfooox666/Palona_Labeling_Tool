"""Xiaolongbao staging-table dwell-time use case."""

DEFAULT_CONFIG = "configs/use_cases/xlb_table_dwell.yaml"
RELEVANT_EVENTS = (
    "order_candidate_appeared",
    "order_confirmed",
    "order_removed",
    "dwell_time_exceeded",
)
