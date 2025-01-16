# -*- coding: utf-8 -*-
"""
to test:
from pure enum inputs
from enum+custom inputs
subsets
exclusions
load
save
add segmentations
"""

# Built-Ins
import pathlib

# Third Party
import pandas as pd
import pytest

# Local Imports
from caf.base import segmentation

# # # CONSTANTS # # #


# # # CLASSES # # #
@pytest.fixture(scope="session", name="vanilla_seg")
def fix_vanilla_segmentation():
    input = segmentation.SegmentationInput(
        enum_segments=["ca", "m", "gender_3"],
        naming_order=["ca", "m", "gender_3"],
    )
    return segmentation.Segmentation(input)


@pytest.fixture(scope="session", name="expected_vanilla_ind")
def fix_exp_vanilla_ind():
    ca = [1, 2]
    g = [1, 2, 3]
    m = [1, 2, 3, 4, 5, 6, 7, 8]
    ind = pd.MultiIndex.from_product([ca, m, g], names=["ca", "m", "gender_3"])
    return ind


@pytest.fixture(scope="session", name="nam_ord_seg")
def fix_nam_ord_seg():
    input = segmentation.SegmentationInput(
        enum_segments=["ca", "m", "gender_3"],
        naming_order=["ca", "gender_3", "m"],
    )
    return segmentation.Segmentation(input)


@pytest.fixture(scope="session", name="exp_nam_ord")
def fix_exp_nam_ord():
    ca = [1, 2]
    g = [1, 2, 3]
    m = [1, 2, 3, 4, 5, 6, 7, 8]
    ind = pd.MultiIndex.from_product([ca, g, m], names=["ca", "gender_3", "m"])
    return ind


@pytest.fixture(scope="session", name="seg_with_excl")
def fix_seg_with_excl():
    conf = segmentation.SegmentationInput(
        enum_segments=["gender_3", "soc", "ca"], naming_order=["gender_3", "soc", "ca"]
    )
    return segmentation.Segmentation(conf)


@pytest.fixture(scope="session", name="expected_excl")
def fix_exp_excl():
    return pd.MultiIndex.from_tuples(
        [
            (1, 4, 1),
            (1, 4, 2),
            (2, 1, 1),
            (2, 1, 2),
            (2, 2, 1),
            (2, 2, 2),
            (2, 3, 1),
            (2, 3, 2),
            (2, 4, 1),
            (2, 4, 2),
            (3, 1, 1),
            (3, 1, 2),
            (3, 2, 1),
            (3, 2, 2),
            (3, 3, 1),
            (3, 3, 2),
            (3, 4, 1),
            (3, 4, 2),
        ]
    )


@pytest.fixture(scope="session", name="subset_seg")
def fix_subset_seg():
    conf = segmentation.SegmentationInput(
        enum_segments=["p", "gender_3", "ns_sec"],
        subsets={"p": list(range(1, 9))},
        naming_order=["p", "gender_3", "ns_sec"],
    )
    return segmentation.Segmentation(conf)


@pytest.fixture(scope="session", name="exp_subset")
def fix_exp_sub():
    p = [1, 2, 3, 4, 5, 6, 7, 8]
    g = [1, 2, 3]
    ns = [1, 2, 3, 4, 5]
    return pd.MultiIndex.from_product([p, g, ns], names=["p", "gender_3", "ns_sec"])


@pytest.fixture(scope="session", name="exp_add")
def fix_add_exp():
    conf = segmentation.SegmentationInput(
        enum_segments=["gender_3", "soc", "ca", "p", "ns_sec"],
        subsets={"p": list(range(1, 9))},
        naming_order=["gender_3", "soc", "ca", "p", "ns_sec"],
    )
    return segmentation.Segmentation(conf)


@pytest.fixture(scope="session", name="simple_segmentation")
def fix_simple_segmentation():
    """Segmentation containing ca and mode only."""
    input = segmentation.SegmentationInput(
        enum_segments=["ca", "m"],
        naming_order=["ca", "m"],
    )
    return segmentation.Segmentation(input)


class TestInd:
    def test_vanilla_ind(self, vanilla_seg, expected_vanilla_ind):
        assert expected_vanilla_ind.equal_levels(vanilla_seg.ind())

    def test_name_order(self, nam_ord_seg, exp_nam_ord):
        assert exp_nam_ord.equal_levels(nam_ord_seg.ind())

    # @pytest.mark.parametrize("segmentation", ["excl_segmentation", "excl_segmentation_rev"])

    def test_exclusions(self, seg_with_excl, expected_excl):
        assert seg_with_excl.ind().equal_levels(expected_excl)

    def test_subset(self, subset_seg, exp_subset):
        assert exp_subset.equal_levels(subset_seg.ind())

    @pytest.mark.parametrize(
        "seg_str", ["subset_seg", "seg_with_excl", "nam_ord_seg", "vanilla_seg"]
    )
    def test_io(self, seg_str, main_dir, request):
        """Check that segmentation objects can be saved and loaded"""
        seg = request.getfixturevalue(seg_str)
        seg.save(main_dir / "meta.yml", "yaml")
        read = segmentation.Segmentation.load(main_dir / "meta.yml", "yaml")
        assert read == seg

    def test_add(self, seg_with_excl, subset_seg, exp_add):
        added = seg_with_excl + subset_seg
        assert added == exp_add

    def test_agg(self, vanilla_seg):
        aggregated = vanilla_seg.aggregate(["ca", "m"])
        conf = segmentation.SegmentationInput(
            enum_segments=["ca", "m"], naming_order=["ca", "m"]
        )
        assert aggregated == segmentation.Segmentation(conf)


class TestSegmentation:
    """General tests for `Segmentation` class."""

    @pytest.mark.parametrize(
        ["segment_params", "expected"],
        [
            ({"ca": 1, "gender_3": 1, "m": 1}, "ca1_m1_gt1"),
            ({"ca": 2, "gender_3": 10, "m": 13}, "ca2_m13_gt10"),
        ],
    )
    def test_generate_segment_name(
        self,
        vanilla_seg: segmentation.Segmentation,
        segment_params: dict[str, int],
        expected: str,
    ):
        """Test `Segmentation.generate_segment_name` produces correct names."""
        name = vanilla_seg.generate_slice_name(segment_params)

        assert name == expected, "incorrect segment name generated"

    @pytest.mark.parametrize("segment_params", [{"ca": 1}, {"gender_3": 1, "m": 2}, {}])
    def test_generate_segment_name_missing(
        self, vanilla_seg: segmentation.Segmentation, segment_params: dict[str, int]
    ):
        """Test `Segmentation.generate_segment_name` correctly raises KeyError.

        KeyError should be raised when `segment_params` doesn't contain all required
        segments.
        """
        error_msg = "missing segments when generating name:"
        with pytest.raises(KeyError, match=error_msg):
            vanilla_seg.generate_slice_name(segment_params)

    @pytest.mark.parametrize(
        ["segment_params", "expected"],
        [
            ({"ca": 1, "gender_3": 1, "m": 1}, (1, 1, 1)),
            ({"ca": 2, "gender_3": 10, "m": 13}, (2, 13, 10)),
        ],
    )
    def test_generate_segment_tuple(
        self,
        vanilla_seg: segmentation.Segmentation,
        segment_params: dict[str, int],
        expected: tuple[int, ...],
    ):
        """Test `Segmentation.generate_segment_tuple` produces correct names."""
        name = vanilla_seg.generate_slice_tuple(segment_params)

        assert name == expected, "incorrect segment tuple generated"

    @pytest.mark.parametrize("segment_params", [{"ca": 1}, {"gender_3": 1, "m": 2}, {}])
    def test_generate_segment_tuple_missing(
        self, vanilla_seg: segmentation.Segmentation, segment_params: dict[str, int]
    ):
        """Test `Segmentation.generate_segment_tuple` correctly raises KeyError.

        KeyError should be raised when `segment_params` doesn't contain all required
        segments.
        """
        error_msg = "missing segments when generating tuple:"
        with pytest.raises(KeyError, match=error_msg):
            vanilla_seg.generate_slice_tuple(segment_params)

    def test_iter_segment_parameters(self, simple_segmentation: segmentation.Segmentation):
        """Test `Segmentation.iter_segment_parameters` produces correct dictionaries."""
        # fmt: off
        expected = [
            {"ca": 1, "m": 1}, {"ca": 2, "m": 1},
            {"ca": 1, "m": 2}, {"ca": 2, "m": 2},
            {"ca": 1, "m": 3}, {"ca": 2, "m": 3},
            {"ca": 1, "m": 4}, {"ca": 2, "m": 4},
            {"ca": 1, "m": 5}, {"ca": 2, "m": 5},
            {"ca": 1, "m": 6}, {"ca": 2, "m": 6},
            {"ca": 1, "m": 7}, {"ca": 2, "m": 7},
            {"ca": 1, "m": 8}, {"ca": 2, "m": 8},
        ]
        # fmt: on
        expected = sorted(expected, key=simple_segmentation.generate_slice_tuple)
        answer = sorted(
            list(simple_segmentation.iter_slices()),
            key=simple_segmentation.generate_slice_tuple,
        )

        assert answer == expected, "incorrect segmentation parameters"

    def test_find_files(self, vanilla_seg: segmentation.Segmentation, tmp_path: pathlib.Path):
        """Test `Segmentation.find_files` finds correct files."""
        folder = tmp_path / "find_files"
        folder.mkdir()

        template = "test_file_{segment_name}"

        expected = []
        for params in vanilla_seg.iter_slices():

            name = vanilla_seg.generate_slice_name(params)
            path = folder / f"{template.format(segment_name=name)}.csv"
            path.touch()
            expected.append(path)

        expected = sorted(expected)

        paths = vanilla_seg.find_files(folder, template, suffixes=[".csv"])
        assert len(paths) == len(vanilla_seg), "incorrect number of files found"

        paths = sorted(list(paths.values()))
        assert paths == expected, "incorrect files found"

    @pytest.mark.parametrize(
        ["tuple_", "expected"],
        [
            ((1, 2, 3), {"ca": 1, "m": 2, "gender_3": 3}),
            ((2, 5, 1), {"ca": 2, "m": 5, "gender_3": 1}),
        ],
    )
    def test_convert_slice_tuple(
        self,
        vanilla_seg: segmentation.Segmentation,
        tuple_: tuple[int, ...],
        expected: dict[str, int],
    ):
        """Test converting a slice tuple into a parameters dictionary."""
        answer = vanilla_seg.convert_slice_tuple(tuple_)
        assert answer == expected

    @pytest.mark.parametrize(
        ["name", "expected"],
        [
            ("ca1_m2_gt3", {"ca": 1, "m": 2, "gender_3": 3}),
            ("ca2_m5_gt1", {"ca": 2, "m": 5, "gender_3": 1}),
        ],
    )
    def test_convert_slice_name(
        self,
        vanilla_seg: segmentation.Segmentation,
        name: str,
        expected: dict[str, int],
    ):
        """Test converting a slice name into a parameters dictionary."""
        answer = vanilla_seg.convert_slice_name(name)
        assert answer == expected
