# -*- coding: utf-8 -*-
"""
Created on: 08/09/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.base import segments

# # # CONSTANTS # # #


# # # CLASSES # # #
@pytest.fixture(scope="session", name="multi-index")
def fix_mult():
    # Define the index levels
    level_a = ["A", "B", "C", "D", "E", "F"]
    level_b = ["G", "H", "I", "J", "K", "L"]
    level_c = ["M", "N", "O", "P", "Q", "R"]
    level_d = ["S", "T", "U", "V", "W", "X"]

    # Create a MultiIndex
    index = pd.MultiIndex.from_tuples(
        [(a, b, c, d) for a, b, c, d in zip(level_a, level_b, level_c, level_d)],
        names=["a", "b", "c", "d"],
    )

    # Create a DataFrame with random data
    data = np.random.rand(6, 1)

    df = pd.DataFrame(data, index=index, columns=["RandomData"])


@pytest.fixture(scope="session", name="expected_excl_ind")
def fix_excl_ind():
    return pd.MultiIndex.from_tuples(
        [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)],
        names=["test seg 1", "test seg 2"],
    )


@pytest.fixture(scope="session", name="get_gender_seg")
def fix_gender_seg():
    return segments.SegmentsSuper("gender_3").get_segment()


@pytest.fixture(scope="session", name="exp_gender_seg")
def fix_exp_gen():
    return segments.Segment(
        name="gender_3",
        values={1: "Child", 2: "Male", 3: "Female"},
        exclusions=[
            segments.Exclusion(
                other_name=segments.SegmentsSuper.SOC.value, exclusions={1: [1, 2, 3]}
            )
        ],
    )


@pytest.fixture(scope="session", name="get_hb_purpose")
def fix_hb_purpose():
    return segments.SegmentsSuper("p").get_segment(subset=list(range(1, 9)))


@pytest.fixture(scope="session", name="expected_hb_purpose")
def fix_exp_hb_purpose():
    return segments.Segment(
        name="p",
        values={
            1: "HB Work",
            2: "HB Employers Business (EB)",
            3: "HB Education",
            4: "HB Shopping",
            5: "HB Personal Business (PB)",
            6: "HB Recreation / Social",
            7: "HB Visiting friends and relatives",
            8: "HB Holiday / Day trip",
        },
    )


class TestSegmentsSuper:
    def test_get(self, get_gender_seg, exp_gender_seg):
        assert get_gender_seg.values == exp_gender_seg.values

    def test_get_subset(self, get_hb_purpose, expected_hb_purpose):
        assert get_hb_purpose == expected_hb_purpose


##### Tests & Fixtures for `Segment` #####


class TestSegment:
    """Tests for the `Segment` class."""

    @pytest.mark.parametrize(
        ["segment", "alias"],
        [
            (segments.SegmentsSuper.DIRECTION, "pa"),
            (segments.SegmentsSuper.ADULTS, "adults"),
        ],
    )
    def test_get_alias(self, segment: segments.SegmentsSuper, alias: str) -> None:
        """Test `get_alias` for segments with and without aliases."""
        seg = segment.get_segment()
        assert seg.get_alias() == alias

    @pytest.mark.parametrize(
        ["segment", "value", "alias"],
        [
            (segments.SegmentsSuper.DIRECTION, 0, "nhb"),
            (segments.SegmentsSuper.GENDER_3, 1, "gt1"),
            (segments.SegmentsSuper.ADULTS, 3, "adults3"),
        ],
    )
    def test_get_value_alias(
        self, segment: segments.SegmentsSuper, value: int, alias: str
    ) -> None:
        """Test `get_value_alias` for segments with and without aliases.

        Ran for all combinations of segments with / without name alias
        and with / without values aliases.
        """
        seg = segment.get_segment()
        assert seg.get_value_alias(value) == alias

    @pytest.mark.parametrize(["segment", "expected"], [
        (segments.SegmentsSuper.ADULTS, r"(?:\b|_)adults(\d+)(?=\b|_)"),
        (segments.SegmentsSuper.GENDER_3, r"(?:\b|_)(?:gender_3|gt)(\d+)(?=\b|_)"),
        (segments.SegmentsSuper.DIRECTION, r"(?:\b|_)(?:(?:direction|pa)(\d+)|(nhb|hb))(?=\b|_)"),
    ])
    def test_value_regex(self, segment: segments.SegmentsSuper, expected: str) -> None:
        """Test `value_regex` returns the correct pattern text."""
        seg = segment.get_segment()
        assert seg.value_regex() == expected

    @pytest.mark.parametrize(
        ["segment", "text", "expected"],
        [
            (segments.SegmentsSuper.DIRECTION, "nhb_hb_pa1_direction0", [0, 1, 1, 0]),
            (segments.SegmentsSuper.PURPOSE, "p1_testing_p7_p12_purpose15", [1, 7, 12]),
            (segments.SegmentsSuper.ADULTS, "something_adults3_adults1", [3, 1]),
        ],
    )
    def test_parse_values(
        self, segment: segments.SegmentsSuper, text: str, expected: list[int]
    ) -> None:
        """Test `extract_values` for segments with / without aliases."""
        seg = segment.get_segment()
        values = seg.extract_values(text)
        assert values == expected
