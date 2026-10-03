import numpy as np
import pytest

from spore_size.summary import describe, format_table, q_systematic, summary_rows


def _rows(image, lengths, widths):
    return [{"image": image, "length_um": l, "width_um": w, "Q_ratio": l / w}
            for l, w in zip(lengths, widths)]


def test_describe_uses_sample_sd_and_sem():
    d = describe([6.0, 7.0, 8.0])
    assert d["n"] == 3
    assert d["mean"] == pytest.approx(7.0)
    assert d["sd"] == pytest.approx(1.0)              # n-1 denominator
    assert d["sem"] == pytest.approx(1.0 / np.sqrt(3))  # SD / sqrt(n)


def test_describe_single_value_has_no_spread():
    d = describe([6.5])
    assert d["mean"] == 6.5 and np.isnan(d["sd"]) and np.isnan(d["sem"])


def test_summary_groups_images_directories_and_all():
    rows = (_rows("a/p1.jpg", [6, 7], [4, 5]) + _rows("a/p2.jpg", [8], [5])
            + _rows("b/p3.jpg", [7, 7, 7], [4, 4, 4]))
    out = summary_rows(rows, images=["a/p1.jpg", "a/p2.jpg", "a/empty.jpg", "b/p3.jpg"])
    assert [(r["level"], r["name"]) for r in out] == [
        ("image", "a/p1.jpg"), ("image", "a/p2.jpg"), ("image", "a/empty.jpg"),
        ("directory", "a"), ("image", "b/p3.jpg"), ("directory", "b"), ("all", "all")]
    by_name = {r["name"]: r for r in out}
    assert by_name["a"]["n_spores"] == 3 and by_name["a"]["n_images"] == 2
    assert by_name["a"]["length_mean"] == pytest.approx(7.0)  # pooled spores
    assert by_name["a/empty.jpg"]["n_spores"] == 0
    assert by_name["all"]["n_spores"] == 6
    assert "all (total)" in format_table(out)


def test_no_directory_rows_without_directories():
    out = summary_rows(_rows("p1.jpg", [6, 7], [4, 5]))
    assert [r["level"] for r in out] == ["image", "all"]


def test_systematic_error_columns():
    rows = _rows("p1.jpg", [6.0, 7.0], [4.0, 5.0])
    out = summary_rows(rows, sys_error_um=0.1)
    r = out[-1]
    assert r["length_sys"] == 0.1 and r["width_sys"] == 0.1
    # Q = 6.5/4.5; independent 0.1 um errors on L and W
    expected = (6.5 / 4.5) * np.hypot(0.1 / 6.5, 0.1 / 4.5)
    assert r["Q_sys"] == pytest.approx(expected)
    assert q_systematic(6.5, 4.5, 0.1) == pytest.approx(expected)
    assert "size" not in r
