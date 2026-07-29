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
import pathlib

# Third Party
import pandas as pd
import pytest

# Local Imports
from caf.base import segments

# # # CONSTANTS # # #


# # # CLASSES # # #
@pytest.fixture(scope="session", name="expected_excl_ind")
def fix_excl_ind():
    return pd.MultiIndex.from_tuples(
        [
            (1, 1),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 1),
            (4, 2),
            (4, 3),
        ],
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
        description="Travel purpose, based on those from the UK National Trip End Model.",
    )


class TestSegmentsSuper:
    def test_get(self, get_gender_seg, exp_gender_seg):
        assert get_gender_seg.values == exp_gender_seg.values

    def test_get_subset(self, get_hb_purpose, expected_hb_purpose):
        assert get_hb_purpose == expected_hb_purpose

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("p", segments.SegmentsSuper.PURPOSE),
            ("UserClass ", segments.SegmentsSuper.USERCLASS),
            ("Age   11 ", segments.SegmentsSuper.AGE_11),
        ],
    )
    def test_valid_init(self, name: str, expected: segments.SegmentsSuper):
        """Test getting segment from valid strings."""
        assert segments.SegmentsSuper(name) == expected

    @pytest.mark.parametrize("name", ["non-existent segment"])
    def test_invalid_init(self, name: str):
        """Test getting ValueError from invalid strings."""
        msg = f"invalid SegmentsSuper: {name!r}"
        with pytest.raises(ValueError, match=msg):
            segments.SegmentsSuper(name)

    def test_get_all(self):
        """Test all segments in :class:`SegmentsSuper` have YAML files."""
        for i in segments.SegmentsSuper:
            i.get_segment()

    def test_extra_yaml(self):
        """Test if any segment YAMLs aren't defined in :class:`SegmentsSuper`."""
        directory = pathlib.Path(segments.__file__).parent / "segments"
        for path in directory.glob("*.yml"):
            segments.SegmentsSuper(path.stem)

    def test_values_returns_all_enum_values(self) -> None:
        """Test `SegmentsSuper.values` mirrors enum `.value` members."""
        expected = [member.value for member in segments.SegmentsSuper]
        assert segments.SegmentsSuper.values() == expected


class TestSegConverter:
    """Tests for `SegConverter` conversion definitions."""

    @pytest.mark.parametrize(
        ["converter", "expected_cols", "expected_levels"],
        [
            (segments.SegConverter.AG_G, ["gender_3"], ["age_9", "g"]),
            (segments.SegConverter.APOPEMP_AWS, ["aws"], ["age_9", "pop_emp"]),
            (
                segments.SegConverter.CARADULT_HHTYPE,
                ["hh_type"],
                ["adults", "car_availability"],
            ),
            (segments.SegConverter.NSSEC_ADULT, ["adult_nssec"], ["ns_sec", "adults"]),
        ],
    )
    def test_get_conversion_structure(
        self,
        converter: segments.SegConverter,
        expected_cols: list[str],
        expected_levels: list[str],
    ) -> None:
        """Test conversion outputs have expected index levels and output columns."""
        out = converter.get_conversion()

        assert list(out.columns) == expected_cols
        assert list(out.index.names) == expected_levels
        assert len(out) > 0

    def test_get_conversion_invalid_input(self) -> None:
        """Test invalid conversion input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input segment"):
            segments.SegConverter.get_conversion("not_a_converter")


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

    @pytest.mark.parametrize(
        ["segment", "expected"],
        [
            (segments.SegmentsSuper.ADULTS, r"(?:\b|_)adults(\d+)(?=\b|_)"),
            (segments.SegmentsSuper.GENDER_3, r"(?:\b|_)(?:gender_3|gt)(\d+)(?=\b|_)"),
            (
                segments.SegmentsSuper.DIRECTION,
                r"(?:\b|_)(?:(?:direction|pa)(\d+)|(nhb|hb))(?=\b|_)",
            ),
        ],
    )
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

    def test_get_value_alias_invalid_value(self) -> None:
        """Test invalid segment value raises a clear error."""
        seg = segments.SegmentsSuper.GENDER_3.get_segment()
        with pytest.raises(ValueError, match="invalid value"):
            seg.get_value_alias(999)

    def test_exclusion_and_lookup_indices(self) -> None:
        """Test exclusion and lookup helper methods return expected indices."""
        corr = segments.Exclusion(other_name="target", exclusions={1: [2, 3], 2: [1]})
        seg = segments.Segment(
            name="source",
            values={1: "A", 2: "B"},
            exclusions=[corr],
            lookups=[corr],
        )

        expected = pd.MultiIndex.from_tuples(
            [(1, 2), (1, 3), (2, 1)], names=["dummy", "target"]
        )

        assert corr.build_index().equals(expected)
        assert seg.drop_indices("target").equals(expected)
        assert seg.lookup_indices("target").equals(expected)
        assert seg.drop_indices("other") is None
        assert seg.lookup_indices("other") is None

    def test_translate_segment_invalid_type(self) -> None:
        """Test translate_segment rejects unsupported input type."""
        seg = segments.SegmentsSuper.AGE_11.get_segment()

        with pytest.raises(TypeError, match="expects either"):
            seg.translate_segment(123)

    def test_val_to_int(self) -> None:
        """Test `val_to_int` inverts the `values` mapping."""
        seg = segments.SegmentsSuper.GENDER_3.get_segment()
        expected = {value: key for key, value in seg.values.items()}

        assert seg.val_to_int == expected

    def test_int_values_len_and_get_alias_without_alias(self) -> None:
        """Test simple Segment property helpers on a custom segment."""
        seg = segments.Segment(name="custom", values={1: "a", 2: "b"})

        assert seg.int_values == [1, 2]
        assert len(seg) == 2
        assert seg.get_alias() == "custom"

    def test_translate_segment_reverse(self) -> None:
        """Test reverse segment lookup translation returns expected mapping."""
        seg = segments.SegmentsSuper.AGE_NTEM.get_segment()
        new_seg, lookup = seg.translate_segment(
            segments.SegmentsSuper.AGE_11, reverse=True
        )

        assert new_seg.name == "age_11"
        assert lookup.name == "age_11"
        assert list(lookup.loc[1]) == [1, 2, 3]
        assert list(lookup.loc[2]) == [4, 5, 6, 7, 8, 9]
        assert list(lookup.loc[3]) == [10, 11]

    def test_add_corr_from_df(self) -> None:
        """Test adding lookup and exclusion correlations from conversion files."""
        seg = segments.SegmentsSuper.AGE_11.get_segment().copy()
        seg.lookups = []
        seg.exclusions = []

        seg.add_corr_from_df(segments.SegmentsSuper.AGE_NTEM, exclusion=False)
        seg.add_corr_from_df(segments.SegmentsSuper.AGE_NTEM, exclusion=True)

        assert len(seg.lookups) == 1
        assert seg.lookups[0].other_name == "age_ntem"
        assert len(seg.exclusions) == 1
        assert seg.exclusions[0].other_name == "age_ntem"
