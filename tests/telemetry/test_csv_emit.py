import io
from telemetry.aggregator import AggregatedMetrics
from telemetry.csv import write_aggregates_csv


def _mk_agg(idx: int) -> AggregatedMetrics:
    # Create simple deterministic aggregates
    start = idx * 10
    end = start + 10
    return AggregatedMetrics(
        cycles=10,
        start_cycle=start,
        end_cycle=end,
        produced_flits_total=100 + idx,
        flits_tx_main=100 + idx,
        flits_tx_esc=0,
        xbar_switches=0,
        mc_served_requests=5 * idx,
        mc_avg_latency=200.0 + idx if idx % 2 == 0 else None,
        mc_p95_latency=500.0 + idx if idx % 2 == 0 else None,
        power_total_energy=50.0 + idx,
        power_avg_power=5.0 + idx / 10.0,
        max_temp_any_tile=60.0 + idx,
        avg_temp_all_tiles=40.0 + idx / 2.0,
        queue_depth_p95_proxy=7.0 + idx / 3.0,
    )


def test_write_aggregates_csv_basic():
    aggs = [_mk_agg(0), _mk_agg(1), _mk_agg(2)]
    buf = io.StringIO()
    nrows = write_aggregates_csv(aggs, buf)
    assert nrows == 3

    csv_text = buf.getvalue().strip().splitlines()
    # Header plus 3 rows
    assert len(csv_text) == 1 + 3

    header = csv_text[0].split(",")
    expected_header = [
        "cycles",
        "start_cycle",
        "end_cycle",
        "produced_flits_total",
        "flits_tx_main",
        "flits_tx_esc",
        "xbar_switches",
        "mc_served_requests",
        "mc_avg_latency",
        "mc_p95_latency",
        "power_total_energy",
        "power_avg_power",
        "max_temp_any_tile",
        "avg_temp_all_tiles",
        "queue_depth_p95_proxy",
    ]
    assert header == expected_header

    # Check formatting invariants on first row (floats have 6 decimals, None -> empty)
    row1 = csv_text[1].split(",")
    # cycles, start, end are ints as strings
    assert row1[0] == "10"
    # mc_avg_latency for idx=0 is present and formatted to 6 decimals
    assert row1[8] == "200.000000"
    # For idx=1 in row2, mc_avg_latency should be empty
    row2 = csv_text[2].split(",")
    assert row2[8] == ""
    # Float formatting fixed precision
    assert row1[10].count(".") == 1
    assert row1[10].split(".")[1] == "000000"