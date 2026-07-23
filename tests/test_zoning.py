# -*- coding: utf-8 -*-
"""Tests for the `ZoningSystem` class."""

# Built-Ins
import dataclasses
import string
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

# Local Imports
from caf.base.zoning import (
    TranslationError,
    TranslationWeighting,
    ZoningSystem,
    ZoningSystemMetaData,
)


@dataclasses.dataclass
class ZoningData:
    """Zoning system dataset."""

    name: str
    data: pd.DataFrame
    subsets: dict[str, list[int]]


@pytest.fixture(name="zoning_data", scope="module")
def fix_zoning_data() -> ZoningData:
    """Define zoning input dataset."""
    ids = list(range(10))
    names = list(string.ascii_lowercase[:10])
    zones = pd.DataFrame(
        {
            "zone_id": ids,
            "zone_name": names,
            "descriptions": [f"{i}-{j}" for i, j in zip(ids, names)],
            "internal": [i < 5 for i in ids],
            "external": [i >= 5 for i in ids],
            "north": [i < 3 for i in ids],
        }
    )

    return ZoningData(
        name="test_zoning",
        data=zones,
        subsets={
            "internal": [0, 1, 2, 3, 4],
            "external": [5, 6, 7, 8, 9],
            "north": [0, 1, 2],
        },
    )


@pytest.fixture(name="old_zoning_dir")
def fix_old_zoning_dir(zoning_data: ZoningData, tmp_path: Path) -> Path:
    """Save zoning data in old format and return directory path."""
    zoning_dir = tmp_path / zoning_data.name
    zoning_dir.mkdir()

    zones = zoning_data.data[["zone_id", "zone_name", "descriptions"]].copy()
    zones = zones.rename(columns={"descriptions": "zone_desc"})
    zones.to_csv(zoning_dir / "zones.csv.bz2", index=False)

    for subset in zoning_data.subsets:
        data = zoning_data.data.loc[zoning_data.data[subset], "zone_id"].to_frame()
        data.to_csv(zoning_dir / f"{subset}_zones.csv.bz2", index=False)

    return zoning_dir


@pytest.fixture(name="id_only_zoning", scope="module")
def fix_id_only_zoning(zoning_data: ZoningData) -> tuple[ZoningData, ZoningSystem]:
    """Create `ZoningSystem` class containing only zone ID."""
    data = ZoningData(
        name=zoning_data.name + "-id_only",
        data=zoning_data.data["zone_id"].to_frame().copy(),
        subsets={},
    )
    system = ZoningSystem(
        name=data.name,
        unique_zones=data.data.copy(),
        metadata=ZoningSystemMetaData(name=data.name),
    )

    return data, system


@pytest.fixture(name="zoning_descriptions", scope="module")
def fix_zoning_descriptions(zoning_data: ZoningData) -> tuple[ZoningData, ZoningSystem]:
    """Create `ZoningSystem` class with optional name and description columns."""
    data = ZoningData(
        name=zoning_data.name + "-zoning_descriptions",
        data=zoning_data.data[["zone_id", "zone_name", "descriptions"]].copy(),
        subsets={},
    )
    system = ZoningSystem(
        name=data.name,
        unique_zones=data.data.copy(),
        metadata=ZoningSystemMetaData(name=data.name),
    )

    return data, system


@pytest.fixture(name="zoning_subsets", scope="module")
def fix_zoning_subsets(zoning_data: ZoningData) -> tuple[ZoningData, ZoningSystem]:
    """ZoningSystem containing all columns including some subsets."""
    data = ZoningData(
        name=zoning_data.name + "-zoning_subsets",
        data=zoning_data.data.copy(),
        subsets=zoning_data.subsets.copy(),
    )
    system = ZoningSystem(
        name=data.name,
        unique_zones=data.data.copy(),
        metadata=ZoningSystemMetaData(name=data.name, extra_columns=list(data.subsets)),
    )

    return data, system


class TestZoning:
    """Tests for the `ZoningSystem` class."""

    @pytest.mark.parametrize(
        "columns", [["zone_id"], ["zone_id", "zone_name", "descriptions"]]
    )
    @pytest.mark.parametrize("subset", [True, False])
    def test_init(
        self, zoning_data: ZoningData, columns: list[str], subset: bool
    ) -> None:
        """Test initialising `ZoningSystem` with / without subsets."""
        all_columns = columns.copy()
        if subset:
            all_columns += list(zoning_data.subsets)
        data = (
            zoning_data.data[all_columns]
            .copy()
            .sort_index(axis=0, inplace=False)
            .sort_index(axis=1, inplace=False)
        )

        system = ZoningSystem(
            name=zoning_data.name,
            unique_zones=data,
            metadata=ZoningSystemMetaData(
                name=zoning_data.name,
                extra_columns=list(zoning_data.subsets) if subset else [],
            ),
        )

        data = data.set_index("zone_id")

        assert_array_equal(data.index.values, system.zone_ids, "incorrect zone IDs")
        assert_frame_equal(data, system.zones_data)

        if subset:
            assert sorted(system.subset_columns) == sorted(zoning_data.subsets), (
                "incorrect subsets"
            )

        if "zone_name" in columns:
            assert_series_equal(data["zone_name"], system.zone_names())
        if "descriptions" in columns:
            assert_series_equal(data["descriptions"], system.zone_descriptions())

    # def test_init_errors(self, zoning_data: ZoningData) -> None:
    #     """Test initialising ZoningSystem with invalid, or missing, ID column."""
    #     meta = ZoningSystemMetaData(name=zoning_data.name)
    #     # Test missing zone ID column
    #     with pytest.raises(
    #         ValueError, match=r"mandatory ID column \(zone_id\) missing from zones data"
    #     ):
    #         ZoningSystem(
    #             name=zoning_data.name,
    #             unique_zones=pd.DataFrame({"missing": [1, 2, 3]}),
    #             metadata=meta,
    #         )
    #
    #     # Test non-int ID column
    #     with pytest.raises(ValueError, match=r"zone IDs should be integers not object"):
    #         ZoningSystem(
    #             name=zoning_data.name,
    #             unique_zones=pd.DataFrame({"zone_id": ["incorrect", "type"]}),
    #             metadata=meta,
    #         )
    #
    #     # Test duplicate IDs
    #     with pytest.raises(ValueError, match=r"duplicate zone IDs: 1"):
    #         ZoningSystem(
    #             name=zoning_data.name,
    #             unique_zones=pd.DataFrame({"zone_id": [1, 1, 2, 3]}),
    #             metadata=meta,
    #         )

    def test_init_subsets(self, zoning_data: ZoningData) -> None:
        """Test initialising invalid and valid subset columns."""
        meta = ZoningSystemMetaData(name=zoning_data.name)
        # Test subset type conversions
        data = pd.DataFrame(
            {
                "zone_id": [1, 2, 3],
                "str_int": ["1", "0", "1"],
                "str_bool": ["TRUE", "FALSE", "TRUE"],
            }
        )
        system = ZoningSystem(name=meta.name, unique_zones=data, metadata=meta)

        subset = np.array([1, 3])
        assert_array_equal(subset, system.get_subset("str_int"))
        assert_array_equal(subset, system.get_subset("str_bool"))

        # Test invalid subset types
        data = pd.DataFrame(
            {
                "zone_id": [1, 2, 3],
                "invalid_int": [5, 6, 7],
                "invalid_str": ["invalid", "text", "subset"],
            }
        )
        with pytest.raises(
            ValueError,
            match=r"2 subset columns found which don't contain boolean values:",
        ):
            ZoningSystem(name=meta.name, unique_zones=data, metadata=meta)

    def test_get_subset(self, zoning_subsets: tuple[ZoningData, ZoningSystem]) -> None:
        """Test getting valid subsets and the various errors."""
        data, system = zoning_subsets
        system = system.copy()

        for name, values in data.subsets.items():
            subset = system.get_subset(name)
            assert_array_equal(
                np.array(values), subset, f"incorrect values for subset {name}"
            )

            subset = system.get_inverse_subset(name)
            assert_array_equal(
                np.array([i for i in data.data["zone_id"] if i not in values]), subset
            )

        with pytest.raises(KeyError):
            system.get_subset("subset_that_doesn't_exist")

        with pytest.raises(ValueError):
            system.get_subset(system._id_column)

        # This shouldn't be done during normal use
        name = "incorrect_dummy_subset"
        system._zones.loc[:, name] = 5
        with pytest.raises(TypeError):
            system.get_subset(name)

    def test_old_to_new_zoning(
        self, old_zoning_dir: Path, zoning_data: ZoningData
    ) -> None:
        """Test `old_to_new_zoning` method can load in old format and output new."""
        new_dir = old_zoning_dir / "new"
        new_dir.mkdir(exist_ok=True)
        zones = ZoningSystem.old_to_new_zoning(old_zoning_dir, new_dir=new_dir)

        expected = ZoningSystem(
            zoning_data.name,
            zoning_data.data,
            metadata=ZoningSystemMetaData(name=zoning_data.name),
        )
        assert zones == expected, "old zoning data loaded isn't as expected"

        new_dir = new_dir / zoning_data.name
        assert new_dir.is_dir(), "new zoning folder not created"

        out_data = pd.read_csv(new_dir / "zoning.csv")
        for i in (0, 1):
            out_data = out_data.sort_index(axis=i)
            zoning_data.data = zoning_data.data.sort_index(axis=i)

        assert_frame_equal(out_data, zoning_data.data)

        out_meta = ZoningSystemMetaData.load_yaml(new_dir / "zoning_meta.yml")
        assert out_meta == expected.metadata, "incorrect metadata"

    @pytest.mark.parametrize(
        "zone_system_str",
        ["id_only_zoning", "zoning_descriptions", "zoning_subsets"],
    )
    def test_io(self, zone_system_str, main_dir, request) -> None:
        """Test saving and loading ZoningSystem's to / from CSVs.

        HDF I/O makes more sense to be tested with DVec.
        """
        zone_system: ZoningSystem
        _, zone_system = request.getfixturevalue(zone_system_str)
        zone_system.save(main_dir, "csv")
        in_zoning = ZoningSystem.load(main_dir / zone_system.name, "csv")
        assert in_zoning == zone_system, "zone system not equal after save then load"

    def test_zone_trans(
        self,
        test_trans: pd.DataFrame,
        min_zoning_2: ZoningSystem,
        min_zoning: ZoningSystem,
        main_dir: Path,
    ):
        """Test sucessfully obtaining zone_translation data."""
        trans = min_zoning_2.translate(min_zoning, cache_path=main_dir)
        assert trans.vector.equals(
            test_trans[["zone_2_id", "zone_1_id", "zone_2_to_zone_1"]]
        )
        assert min_zoning_2.translation_column_name(min_zoning) == "zone_2_to_zone_1"

    def test_getter(self, id_only_zoning: tuple[ZoningData, ZoningSystem], main_dir):
        """Test finding a zone system based on name."""
        _, system = id_only_zoning
        system.save(main_dir, "csv")
        got_zone = ZoningSystem.get_zoning(system.name, search_dir=main_dir)
        assert got_zone == system, "zoning system not equal after load"

    def test_translation_weighting_suffixes(self) -> None:
        """Test all translation weightings map to expected filename suffixes."""
        expected = {
            TranslationWeighting.SPATIAL: "spatial",
            TranslationWeighting.POPULATION: "population_weight",
            TranslationWeighting.EMPLOYMENT: "employment_weight",
            TranslationWeighting.NO_WEIGHT: "no_weighting",
            TranslationWeighting.AVERAGE: "weighted_average",
            TranslationWeighting.POP: "pop",
            TranslationWeighting.EMP: "emp",
        }
        for weighting, suffix in expected.items():
            assert weighting.get_suffix() == suffix

    def test_lookup_properties(
        self, zoning_subsets: tuple[ZoningData, ZoningSystem]
    ) -> None:
        """Test ID/name/description lookup helper properties."""
        _, system = zoning_subsets

        assert system.name_to_id["a"] == 0
        assert system.id_to_name[0] == "a"
        assert system.desc_to_id["0-a"] == 0
        assert system.id_to_desc[0] == "0-a"
        assert system.id_to_internal[0]
        assert not system.id_to_external[0]

    def test_load_invalid_mode(self, main_dir: Path) -> None:
        """Test loading with unsupported mode errors."""
        with pytest.raises(ValueError, match="Mode can only be"):
            ZoningSystem.load(main_dir, "invalid")

    def test_zoning_from_df_col(self) -> None:
        """Test creating a zoning system from a dataframe column."""
        col = pd.Series(["A", "B", "A"], name="test_col")
        zoning = ZoningSystem.zoning_from_df_col(col)

        assert zoning.name == "test_col"
        assert set(zoning.zone_ids) == {"A", "B"}

    def test_trans_df_to_dict(self, test_trans: pd.DataFrame) -> None:
        """Test conversion of nested translation dataframe to dict."""
        out = ZoningSystem.trans_df_to_dict(
            test_trans,
            from_col="zone_1_id",
            to_col="zone_2_id",
            factor_col="zone_1_to_zone_2",
        )
        assert out[1] == 1
        assert out[5] == 4

    def test_trans_df_to_dict_not_nested(self, test_trans: pd.DataFrame) -> None:
        """Test non-nested translation dataframe raises TranslationError."""
        with pytest.raises(TranslationError, match="nested zoning systems"):
            ZoningSystem.trans_df_to_dict(
                test_trans,
                from_col="zone_2_id",
                to_col="zone_1_id",
                factor_col="zone_2_to_zone_1",
            )

    def test_validate_translation_missing_columns(
        self, min_zoning: ZoningSystem, min_zoning_2: ZoningSystem
    ) -> None:
        """Test translation validation fails when required columns are missing."""
        bad = pd.DataFrame(
            {
                min_zoning.column_name: [1, 2, 3],
                min_zoning_2.column_name: [1, 2, 3],
            }
        )

        with pytest.raises(TranslationError, match="required columns missing"):
            min_zoning.validate_translation_data(min_zoning_2, bad)

    def test_translate_invalid_other_type(self, min_zoning: ZoningSystem) -> None:
        """Test translating with invalid target type raises ValueError."""
        with pytest.raises(ValueError, match="Expected ZoningSystem"):
            min_zoning.translate("not-a-zoning")

    def test_get_translation_definition_missing_weighting_file(
        self,
        min_zoning: ZoningSystem,
        min_zoning_2: ZoningSystem,
        main_dir: Path,
        test_trans: pd.DataFrame,
    ) -> None:
        """Test missing weighting file in existing cache folder raises TranslationError."""
        _ = test_trans
        with pytest.raises(TranslationError, match="different weighting"):
            min_zoning._get_translation_definition(
                min_zoning_2,
                weighting=TranslationWeighting.POPULATION,
                trans_cache=main_dir,
            )

    def test_check_all_columns(self, zoning_subsets: tuple[ZoningData, ZoningSystem]) -> None:
        """Test helper for selecting best ID replacement mapping from input columns."""
        _, system = zoning_subsets

        assert system.check_all_columns(pd.Series(system.zone_ids)) is None
        lookup = system.check_all_columns(pd.Series(system.zone_names().values))
        assert lookup == system.name_to_id
