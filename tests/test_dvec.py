# -*- coding: utf-8 -*-
"""
To test:

build from dataframe
build from old format
save
load
add
subtract
mul
div
aggregate
translate

"""

# Built-Ins
from math import isclose

# Third Party
import numpy as np
import pandas as pd
import pytest

# Local Imports
from caf.base import data_structures, segmentation
from caf.base.segments import SegmentsSuper

# # # CONSTANTS # # #


@pytest.fixture(name="dvec_data_1", scope="session")
def fix_data_1(basic_segmentation_1, min_zoning):
    df = pd.DataFrame(
        data=np.random.rand(24, 5),
        index=basic_segmentation_1.ind(),
        columns=min_zoning.zone_ids,
    )
    df.columns.name = "zone_1_id"
    return df


@pytest.fixture(name="dvec_data_2", scope="session")
def fix_data_2(basic_segmentation_2, min_zoning):
    return pd.DataFrame(
        data=np.random.rand(9, 5),
        index=basic_segmentation_2.ind(),
        columns=min_zoning.zone_ids,
    )


@pytest.fixture(name="single_seg_dvec", scope="session")
def fix_single_seg(min_zoning):
    seg_conf = segmentation.SegmentationInput(enum_segments=["p"], naming_order=["p"])
    seg = segmentation.Segmentation(seg_conf)
    data = pd.DataFrame(
        data=np.random.rand(16, 5), index=seg.ind(), columns=min_zoning.zone_ids
    )
    return data_structures.DVector(
        segmentation=seg, import_data=data, zoning_system=min_zoning
    )


@pytest.fixture(name="no_zone_dvec_1", scope="session")
def fix_no_zone_1(basic_segmentation_1):
    data = pd.Series(
        np.random.rand(
            24,
        ),
        index=basic_segmentation_1.ind(),
    )
    return data_structures.DVector(segmentation=basic_segmentation_1, import_data=data)


@pytest.fixture(name="no_zone_dvec_2", scope="session")
def fix_no_zone_2(basic_segmentation_2):
    data = pd.Series(
        np.random.rand(
            9,
        ),
        index=basic_segmentation_2.ind(),
    )
    return data_structures.DVector(segmentation=basic_segmentation_2, import_data=data)


@pytest.fixture(name="basic_dvec_1", scope="session")
def fix_basic_dvec_1(min_zoning, basic_segmentation_1, dvec_data_1):
    return data_structures.DVector(
        segmentation=basic_segmentation_1,
        zoning_system=min_zoning,
        import_data=dvec_data_1,
    )


@pytest.fixture(name="basic_dvec_2", scope="session")
def fix_basic_dvec_2(min_zoning, basic_segmentation_2, dvec_data_2):
    return data_structures.DVector(
        segmentation=basic_segmentation_2,
        zoning_system=min_zoning,
        import_data=dvec_data_2,
    )


# @pytest.fixture(name="")


@pytest.fixture(name="comp_zoned_dvec", scope="session")
def fix_comp_dvec(
    min_zoning, min_zoning_2, test_trans, basic_segmentation_1, dvec_data_1
):
    data = dvec_data_1.mul(
        test_trans.set_index(["zone_1_id", "zone_2_id"])["zone_1_to_zone_2"], axis=1
    )
    data.columns = data.columns.reorder_levels(["zone_2_id", "zone_1_id"])
    return data_structures.DVector(
        segmentation=basic_segmentation_1,
        zoning_system=[min_zoning_2, min_zoning],
        import_data=data,
    )


@pytest.fixture(name="expected_trans", scope="session")
def fix_exp_trans(basic_dvec_1, min_zoning_2):
    orig_data = basic_dvec_1.data
    trans_data = pd.DataFrame(
        index=orig_data.index,
        data={
            1: orig_data[1],
            2: orig_data[2],
            3: orig_data[3],
            4: orig_data[4] + orig_data[5],
        },
    )
    return data_structures.DVector(
        segmentation=basic_dvec_1.segmentation,
        zoning_system=min_zoning_2,
        import_data=trans_data,
    )


# # # CLASSES # # #


# # # FUNCTIONS # # #
class TestDvec:
    def test_comp_zone(self, basic_dvec_1, test_trans, min_zoning_2, comp_zoned_dvec):
        test = basic_dvec_1.composite_zoning(min_zoning_2, test_trans)
        assert test == comp_zoned_dvec

    @pytest.mark.parametrize(
        "dvec", ["basic_dvec_1", "basic_dvec_2", "comp_zoned_dvec"]
    )
    @pytest.mark.parametrize("subset", [None, [1, 2, 3]])
    @pytest.mark.parametrize("method", ["split", "duplicate"])
    def test_add_segments(self, dvec, subset, method, request):
        dvec_arg = request.getfixturevalue(dvec).copy()
        segment = SegmentsSuper("tp").get_segment(subset=subset)
        out_dvec = dvec_arg.add_segments([segment], split_method=method)
        new_seg_len = len(segment)
        if method == "split":
            assert isclose(out_dvec.data.values.sum(), dvec_arg.data.values.sum())
        else:
            assert isclose(
                out_dvec.data.values.sum(), dvec_arg.data.values.sum() * new_seg_len
            )

    def test_add_segment_exclusion(self, basic_dvec_1):
        segment = SegmentsSuper("soc").get_segment()
        out_dvec = basic_dvec_1.add_segments([segment], split_method="split")
        assert isclose(out_dvec.data.values.sum(), basic_dvec_1.data.values.sum())

    @pytest.mark.parametrize(
        "dvec", ["basic_dvec_1", "basic_dvec_2", "single_seg_dvec", "comp_zoned_dvec"]
    )
    def test_io(self, dvec, main_dir, request):
        dvec = request.getfixturevalue(dvec)
        dvec.save(main_dir / "dvector.h5")
        read_dvec = data_structures.DVector.load(main_dir / "dvector.h5")
        assert read_dvec == dvec

    @pytest.mark.parametrize(
        "dvec_1_str",
        [
            "basic_dvec_1",
            "basic_dvec_2",
            "no_zone_dvec_1",
            "no_zone_dvec_2",
            "comp_zoned_dvec",
        ],
    )
    @pytest.mark.parametrize(
        "dvec_2_str",
        ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"],
    )
    def test_add(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 + dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.add(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.add(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    @pytest.mark.parametrize(
        "dvec_1_str",
        [
            "basic_dvec_1",
            "basic_dvec_2",
            "no_zone_dvec_1",
            "no_zone_dvec_2",
            "comp_zoned_dvec",
        ],
    )
    @pytest.mark.parametrize(
        "dvec_2_str",
        ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"],
    )
    def test_sub(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 - dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.sub(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.sub(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    @pytest.mark.parametrize(
        "dvec_1_str",
        [
            "basic_dvec_1",
            "basic_dvec_2",
            "no_zone_dvec_1",
            "no_zone_dvec_2",
            "comp_zoned_dvec",
        ],
    )
    @pytest.mark.parametrize(
        "dvec_2_str",
        ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"],
    )
    def test_mul(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 * dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.mul(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.mul(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    @pytest.mark.parametrize(
        "dvec_1_str",
        [
            "basic_dvec_1",
            "basic_dvec_2",
            "no_zone_dvec_1",
            "no_zone_dvec_2",
            "comp_zoned_dvec",
        ],
    )
    @pytest.mark.parametrize(
        "dvec_2_str",
        ["basic_dvec_1", "basic_dvec_2", "no_zone_dvec_1", "no_zone_dvec_2"],
    )
    def test_div(self, dvec_1_str, dvec_2_str, request):
        dvec_1 = request.getfixturevalue(dvec_1_str)
        dvec_2 = request.getfixturevalue(dvec_2_str)
        added_dvec = dvec_1 / dvec_2
        dvec_1_data = dvec_1.data
        dvec_2_data = dvec_2.data
        try:
            added_df = dvec_1_data.div(dvec_2_data, axis="index")
        except:
            added_df = dvec_2_data.div(dvec_1_data, axis="index")
        if added_df.index.names != added_dvec.segmentation.naming_order:
            added_df.index = added_df.index.reorder_levels(
                added_dvec.segmentation.naming_order
            )
        assert added_dvec.data.sort_index().equals(added_df.sort_index())

    def test_trans(
        self, basic_dvec_1, test_trans, min_zoning_2, expected_trans, main_dir
    ):
        translation = basic_dvec_1.translate_zoning(min_zoning_2, cache_path=main_dir)
        assert translation == expected_trans

    def test_agg(self, basic_dvec_1):
        aggregated = basic_dvec_1.aggregate(["gender_3"])
        grouped = basic_dvec_1.data.groupby(level="gender_3").sum()
        assert grouped.equals(aggregated.data)

    def test_validate_ipf_targets_series(self, basic_dvec_1):
        """Test validating IPF targets provided as zonal totals series."""
        target_series = basic_dvec_1.data.sum(axis=0)
        target = data_structures.IpfTarget(data=target_series)

        validated = basic_dvec_1.validate_ipf_targets([target])

        assert validated[0].data.equals(target_series)

    def test_validate_ipf_targets_series_bad_zoning_name(self, basic_dvec_1):
        """Test series targets with non-matching zoning names are rejected."""
        target_series = basic_dvec_1.data.sum(axis=0).copy()
        target_series.index = target_series.index.rename("bad_zone_name")

        with pytest.raises(
            data_structures.ZoningError, match="Zoning systems do not match"
        ):
            basic_dvec_1.validate_ipf_targets(
                [data_structures.IpfTarget(data=target_series)]
            )

    def test_ipf_with_series_target(self, basic_dvec_1):
        """Test IPF accepts and handles a pd.Series target."""
        target_series = basic_dvec_1.data.sum(axis=0)
        target = data_structures.IpfTarget(data=target_series)

        fitted, rmse = basic_dvec_1.ipf([target], max_iters=2)

        assert isclose(rmse, 0.0, abs_tol=1e-10)
        assert np.allclose(fitted.data.values, basic_dvec_1.data.values)

    def test_balance_protect_subset(self, basic_dvec_1, min_zoning):
        """Test balancing while preserving protected subset values."""
        target = basic_dvec_1.copy()
        target.data = target.data * 1.5

        protected_subset = {"gender_3": [1]}
        balanced = basic_dvec_1.balance_protect_subset(
            target,
            target_zone=min_zoning,
            protected_subset=protected_subset,
        )

        protected_mask = basic_dvec_1.data.index.get_level_values("gender_3").isin([1])
        assert balanced.data.loc[protected_mask].equals(
            basic_dvec_1.data.loc[protected_mask]
        )
        assert np.allclose(
            balanced.data.sum(axis=0).values,
            target.data.sum(axis=0).values,
        )

    def test_rename_segment(self, no_zone_dvec_1):
        """Test `rename_segment` updates segmentation and index names."""
        renamed = no_zone_dvec_1.rename_segment({"m": "mode"})

        assert "mode" in renamed.segmentation.names
        assert "m" not in renamed.segmentation.names
        assert "mode" in renamed.data.index.names
        assert "m" not in renamed.data.index.names
        assert isclose(renamed.data.sum(), no_zone_dvec_1.data.sum())

    def test_remove_zoning_errors(self, basic_dvec_1, no_zone_dvec_1):
        """Test remove_zoning input validation and no-zoning guard."""
        with pytest.raises(ValueError, match="not callable"):
            basic_dvec_1.remove_zoning(fn="not_a_function")

        with pytest.raises(ValueError, match="There is no zoning to remove"):
            no_zone_dvec_1.remove_zoning()

    def test_concat_from_dir_skips_non_dvector_hdf(self, basic_dvec_1, tmp_path):
        """Test concat_from_dir ignores non-DVector hdf files and loads DVector files."""
        basic_dvec_1.save(tmp_path / "valid.hdf")
        pd.DataFrame({"a": [1]}).to_hdf(tmp_path / "not_dvec.hdf", key="data")

        out = data_structures.DVector.concat_from_dir(tmp_path)

        assert out == basic_dvec_1

    def test_concat_from_dir_multizoned_raises(self, comp_zoned_dvec, tmp_path):
        """Test concat_from_dir rejects multizoned DVectors when zoning is inferred."""
        comp_zoned_dvec.save(tmp_path / "comp.hdf")

        with pytest.raises(TypeError, match="singly zoned"):
            data_structures.DVector.concat_from_dir(tmp_path)

    def test_concat_from_dir_segmentation_not_subset_raises(
        self,
        basic_dvec_2,
        basic_segmentation_1,
        min_zoning,
        tmp_path,
    ):
        """Test concat_from_dir errors for non-subset segmentations."""
        basic_dvec_2.save(tmp_path / "seg2.hdf")

        with pytest.raises(data_structures.SegmentationError, match="not a subset"):
            data_structures.DVector.concat_from_dir(
                tmp_path,
                zoning=min_zoning,
                segmentation=basic_segmentation_1,
            )

    def test_concat_and_concat_list(self, basic_dvec_1):
        """Test concat and concat_list with disjoint subset DVectors."""
        subset_1 = basic_dvec_1.filter_segment_values({"gender_3": [1]})
        subset_2 = basic_dvec_1.filter_segment_values({"gender_3": [2, 3]})

        combined = subset_1.concat(subset_2)
        combined_list = data_structures.DVector.concat_list(
            [subset_1, subset_2],
            basic_dvec_1.segmentation,
        )

        assert combined.data.sort_index().equals(basic_dvec_1.data.sort_index())
        assert combined_list.data.sort_index().equals(basic_dvec_1.data.sort_index())

    def test_concat_overlap_and_zoning_errors(self, basic_dvec_1, no_zone_dvec_1):
        """Test concat errors for overlapping indices and mismatched zoning."""
        subset_1 = basic_dvec_1.filter_segment_values({"gender_3": [1]})

        with pytest.raises(ValueError, match="overlap in indices"):
            subset_1.concat(subset_1)

        with pytest.raises(ValueError, match="Zoning systems don't match"):
            basic_dvec_1.concat(no_zone_dvec_1)

    def test_add_value_to_subset(self, basic_dvec_1):
        """Test deprecated add_value_to_subset still appends rows and updates subset."""
        subset_1 = basic_dvec_1.filter_segment_values({"gender_3": [1]})
        data_for_new_value = basic_dvec_1.filter_segment_values({"gender_3": [2]}).data
        data_for_new_value = data_for_new_value.droplevel("gender_3")

        out = subset_1.add_value_to_subset("gender_3", 2, data_for_new_value)

        assert 2 in out.data.index.get_level_values("gender_3")
        assert sorted(out.segmentation.input.subsets["gender_3"]) == [1, 2]

    def test_balance_by_segments_none_and_type_error(
        self, basic_dvec_1, no_zone_dvec_1
    ):
        """Test balancing without zones arg and guard for unzoned inputs."""
        other = basic_dvec_1.copy()
        other.data = other.data * 2

        balanced = basic_dvec_1.balance_by_segments(other, balancing_zones=None)

        assert np.allclose(balanced.data.values, other.data.values)

        with pytest.raises(TypeError, match="single zone systems"):
            no_zone_dvec_1.balance_by_segments(no_zone_dvec_1)


class TestTimeFormat:
    """Tests for `TimeFormat` conversion helpers and validation."""

    def test_get_strips_and_normalises(self):
        """Test `get` accepts mixed-case values with whitespace."""
        out = data_structures.TimeFormat.get("  AVG_DAY  ")
        assert out == data_structures.TimeFormat.AVG_DAY

    def test_get_invalid_raises(self):
        """Test `get` rejects invalid values."""
        with pytest.raises(ValueError, match="time_format is not valid"):
            data_structures.TimeFormat.get("bad_value")

    @pytest.mark.parametrize(
        ["from_", "to_", "expected_fn"],
        [
            (
                data_structures.TimeFormat.AVG_WEEK,
                data_structures.TimeFormat.AVG_DAY,
                data_structures.TimeFormat._week_to_day_factors,
            ),
            (
                data_structures.TimeFormat.AVG_WEEK,
                data_structures.TimeFormat.AVG_HOUR,
                data_structures.TimeFormat._week_to_hour_factors,
            ),
            (
                data_structures.TimeFormat.AVG_DAY,
                data_structures.TimeFormat.AVG_WEEK,
                data_structures.TimeFormat._day_to_week_factors,
            ),
            (
                data_structures.TimeFormat.AVG_DAY,
                data_structures.TimeFormat.AVG_HOUR,
                data_structures.TimeFormat._day_to_hour_factors,
            ),
            (
                data_structures.TimeFormat.AVG_HOUR,
                data_structures.TimeFormat.AVG_WEEK,
                data_structures.TimeFormat._hour_to_week_factors,
            ),
            (
                data_structures.TimeFormat.AVG_HOUR,
                data_structures.TimeFormat.AVG_DAY,
                data_structures.TimeFormat._hour_to_day_factors,
            ),
        ],
    )
    def test_get_conversion_factors(self, from_, to_, expected_fn):
        """Test conversion routing returns factors from the expected helper."""
        out = from_.get_conversion_factors(to_)
        assert out == expected_fn()

    def test_get_conversion_factors_type_validation(self):
        """Test conversion requires TimeFormat destination enum."""
        with pytest.raises(ValueError, match="Expected to_time_format"):
            data_structures.TimeFormat.AVG_WEEK.get_conversion_factors("avg_day")

    def test_get_conversion_factors_same_format_raises(self):
        """Test conversion to self is rejected."""
        with pytest.raises(ValueError, match="converting to self"):
            data_structures.TimeFormat.AVG_DAY.get_conversion_factors(
                data_structures.TimeFormat.AVG_DAY
            )


class TestDVectorValidation:
    """Tests for DVector constructor/setter input validation."""

    def test_constructor_rejects_invalid_zoning_type(self, basic_segmentation_1):
        """Test non-zoning objects for `zoning_system` are rejected."""
        data = pd.Series(np.random.rand(24), index=basic_segmentation_1.ind())

        with pytest.raises(ValueError, match="not a caf.base.ZoningSystem"):
            data_structures.DVector(
                segmentation=basic_segmentation_1,
                import_data=data,
                zoning_system=1,
            )

    def test_constructor_rejects_non_zoning_in_sequence(
        self, basic_segmentation_1, min_zoning
    ):
        """Test sequence zoning input must only contain `ZoningSystem` objects."""
        data = pd.DataFrame(
            np.random.rand(24, 5),
            index=basic_segmentation_1.ind(),
            columns=min_zoning.zone_ids,
        )

        with pytest.raises(TypeError, match="All zoning_systems"):
            data_structures.DVector(
                segmentation=basic_segmentation_1,
                import_data=data,
                zoning_system=[min_zoning, "bad"],
            )

    def test_constructor_rejects_invalid_segmentation_type(self, min_zoning):
        """Test non-segmentation objects are rejected."""
        data = pd.DataFrame(np.random.rand(2, 5), columns=min_zoning.zone_ids)

        with pytest.raises(ValueError, match="not a caf.base.SegmentationLevel"):
            data_structures.DVector(
                segmentation="bad_segmentation",
                import_data=data,
                zoning_system=min_zoning,
            )

    def test_constructor_rejects_unsupported_import_type(self, basic_segmentation_1):
        """Test unsupported `import_data` input types raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Don't know how to deal"):
            data_structures.DVector(
                segmentation=basic_segmentation_1,
                import_data={"not": "a dataframe"},
            )

    def test_data_setter_rejects_non_dataframe_or_series(self, no_zone_dvec_1):
        """Test DVector `data` setter validates pandas input types."""
        with pytest.raises(
            TypeError, match="data must be a pandas DataFrame or Series"
        ):
            no_zone_dvec_1.data = [1, 2, 3]


class TestIpfTarget:
    """Tests for `IpfTarget` validation."""

    def test_multizoned_dvector_rejected(self, comp_zoned_dvec):
        """Test IpfTarget rejects multizoned DVector targets."""
        with pytest.raises(TypeError, match="cannot currently be multizoned"):
            data_structures.IpfTarget(data=comp_zoned_dvec)

    def test_series_allowed(self, basic_dvec_1):
        """Test IpfTarget accepts pd.Series targets."""
        target_series = basic_dvec_1.data.sum(axis=0)
        target = data_structures.IpfTarget(data=target_series)
        assert target.data.equals(target_series)

    def test_check_compatibility_no_adjust(self, basic_dvec_1):
        """Test compatibility check returns RMSE table and unchanged targets."""
        target_1 = basic_dvec_1.copy()
        target_2 = basic_dvec_1.copy()
        target_2.data = target_2.data * 1.1

        rmses, targets_out, differences = data_structures.IpfTarget.check_compatibility(
            [target_1, target_2],
            adjust=False,
            chain_adjust=False,
        )

        assert not rmses.empty
        assert len(targets_out) == 2
        assert len(differences) == 2

    def test_check_compatibility_adjust(self, basic_dvec_1):
        """Test compatibility check can adjust targets towards reference target."""
        target_1 = basic_dvec_1.copy()
        target_1.data = target_1.data * 0.5
        target_2 = basic_dvec_1.copy()

        _, adjusted, _ = data_structures.IpfTarget.check_compatibility(
            [target_1, target_2],
            adjust=True,
            chain_adjust=False,
            reference=target_2,
        )

        assert adjusted[0].sum() > target_1.sum()
        assert isclose(adjusted[1].sum(), target_2.sum())

    def test_check_compatibility_subset_skip(self, basic_dvec_1):
        """Test compatibility loop handles subset targets without crashing."""
        subset_target = basic_dvec_1.filter_segment_values({"gender_3": [1]})
        full_target = basic_dvec_1.copy()

        rmses, targets_out, _ = data_structures.IpfTarget.check_compatibility(
            [subset_target, full_target],
            adjust=False,
            chain_adjust=True,
        )

        assert isinstance(rmses, pd.DataFrame)
        assert len(targets_out) == 2

    def test_validate_ipf_targets_missing_seg_translation_raises(self, basic_dvec_1):
        """Test non-subset target segmentation without translations raises ValueError."""
        target = basic_dvec_1.add_segments(
            [SegmentsSuper("tp").get_segment(subset=[1, 2])], split_method="duplicate"
        )

        with pytest.raises(ValueError, match="segmentation is not a subset"):
            basic_dvec_1.validate_ipf_targets([data_structures.IpfTarget(data=target)])

    def test_validate_ipf_targets_bad_seg_translation_mapping_raises(
        self, basic_dvec_1
    ):
        """Test invalid segment translation mapping raises ValueError."""
        target = basic_dvec_1.add_segments(
            [SegmentsSuper("tp").get_segment(subset=[1, 2])], split_method="duplicate"
        )

        with pytest.raises(ValueError, match="No translation defined"):
            basic_dvec_1.calc_rmse(
                [
                    data_structures.IpfTarget(
                        data=target,
                        segment_translations={"tp": "not_in_seed"},
                    )
                ]
            )
