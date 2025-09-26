# -*- coding: utf-8 -*-
"""
Module for handling segmentation objects.

This imports the Segment class from caf.base.segments, and the SegmentsSuper
enumeration from caf.base.segments. Both are used for building segmentations.
"""
from __future__ import annotations

# Built-Ins
import collections
import collections.abc
import copy
import itertools
import warnings
from os import PathLike
from pathlib import Path
from typing import Iterator, Literal, NamedTuple, Optional, Union

# Third Party
import caf.toolkit as ctk
import h5py
import pandas as pd
import pydantic
from caf.toolkit import BaseConfig

# Local Imports
from caf.base.segments import Segment, SegmentsSuper

# # # CONSTANTS # # #


# # # CLASSES # # #
class SegmentationWarning(Warning):
    """Warn about segmentation objects."""


class SegmentationError(Exception):
    """Error for segmentation objects."""


class SegmentationSlice:
    """Single slice of a segmentation.

    A slice is defined as a single value for each segment in the
    segmentations.

    Parameters
    ----------
    slice_params : dict[str, int]
        Parameters for the slice, where keys are segment names and
        values are the segment values for this slice e.g. {"p": 1, "m": 3}.
    naming_order : collections.abc.Sequence[str] | None, optional
        Optional naming order of the segments in this slice. If not
        provided the order of the keys in `slice_params` will be used.
    """

    def __init__(
        self,
        slice_params: dict[str, int],
        naming_order: collections.abc.Sequence[str] | None = None,
    ):
        """Initialize a SegmentationSlice object from parameters dict.

        Raises
        ------
        ValueError
            If the naming order does not match the slice parameters.
        """
        self._data = dict(slice_params)

        if naming_order is None:
            self._naming_order = tuple(slice_params.keys())
        else:
            self._naming_order = tuple(naming_order)

            names = set(self._naming_order)
            data = set(self._data.keys())
            if names != data:
                raise ValueError(
                    "The naming order provided does not match the keys in the"
                    f" slice parameters. Naming order: {self._naming_order},"
                    f" slice_params: {self._data.keys()}"
                )

    def __getitem__(self, item: str) -> int:
        """Get value of segment for slice."""
        return self._data[item]

    def get(self, item: str, default: int | None = None) -> int | None:
        """Get value of segment for slice with default."""
        return self._data.get(item, default)

    @property
    def naming_order(self) -> tuple[str, ...]:
        """Return the naming order of the segments in this slice."""
        return self._naming_order

    @property
    def data(self) -> dict[str, int]:
        """Return the data of the slice as a dict."""
        return self._data.copy()

    def as_tuple(self) -> tuple[int, ...]:
        """Generate segment tuple from parameters.

        Returns
        -------
        tuple[int, ...]
            Values of slice parameters, the order of the
            parameters is defined by the `naming_order`.
        """
        return tuple(self._data[i] for i in self._naming_order)

    @classmethod
    def from_tuple(
        cls, slice_tuple: tuple[int, ...], naming_order: collections.abc.Sequence[str]
    ) -> "SegmentationSlice":
        """Convert a segmentation slice tuple into parameters.

        Parameters
        ----------
        slice_tuple : tuple[int, ...]
            Tuple containing segment values for a single slice.

        Raises
        ------
        ValueError
            If the tuple given doesn't have the expected number of values.
        """
        if len(slice_tuple) != len(naming_order):
            raise ValueError(
                f"slice tuple should contain {len(naming_order)}"
                f" values but contains {len(slice_tuple)} values"
            )
        return SegmentationSlice(dict(zip(naming_order, slice_tuple)))

    def __hash__(self) -> int:
        """Hash the naming order and values tuples."""
        return hash(self.naming_order + self.as_tuple())

    def __eq__(self, value) -> bool:
        """Check equality of two SegmentationSlice objects."""
        if not isinstance(value, SegmentationSlice):
            return False
        return (self.as_tuple() + self.naming_order) == (value.as_tuple() + value.naming_order)

    def generate_name(self, segments: dict[str, Segment] | None = None) -> str:
        """Generate name for a slice of the segmentation from parameters.

        Parameters
        ----------
        segments
            Dictionary of segments to use for generating the name.
            If not provided, it will attempt to get segments from
            `SegmentsSuper` and if that fails the full segment name
            will be used.

        Returns
        -------
        str
            Name of slice in the form "{name}{value}_{name 2}{value 2}"
            e.g. "p1_m3", where name is the segment alias if it has one.
            For any segments with a value alias that will be used instead
            of "{name}{value}". The order of the parameters is defined by
            the `naming_order`.

        See Also
        --------
        Segment.get_alias: to get the segment alias.
        Segment.get_value_alias
            to get the alias or "{name}{value}" for a specific segment value.
        """
        slice_parts = []
        if segments is None:
            segments = {}

        for name in self.naming_order:
            try:
                segment = segments.get(name)
                if segment is None:
                    segment = SegmentsSuper(name).get_segment()
            except ValueError as exc:
                warnings.warn(
                    f"Could not find segment {name} in segments or"
                    f" SegmentsSuper: {exc}\nUsing full segment name instead.",
                    RuntimeWarning,
                )
                slice_parts.append(f"{name}{self[name]}")
                continue

            slice_parts.append(segment.get_value_alias(self[name]))

        return "_".join(slice_parts)

    def __repr__(self):
        """Get string representation of the slice."""
        params = ", ".join(f"{i}={self[i]}" for i in self.naming_order)
        return f"SegmentationSlice({params})"

    def __contains__(self, item: str) -> bool:
        """Check if segment is in slice."""
        return item in self.data

    def replace(self, current: str, new: str, value: int | None = None) -> "SegmentationSlice":
        """Create a copy of slice with current segment replaced with new.

        Parameters
        ----------
        current, new
            Name of segments.
        value
            Value to set for new segment, if not provided
            uses value of current segment. *Value of current
            segment isn't always valid for new segment.*

        Raises
        ------
        ValueError
            If new segment is already in the slice
            or current segment isn't.
        """
        if new in self._data:
            raise ValueError(f"{new} segment already in slice")
        if current not in self._data:
            raise ValueError(f"{current} segment not in slice")

        new_data = self._data.copy()
        if value is None:
            value = new_data.pop(current)
        else:
            new_data.pop(current)
        new_data[new] = int(value)

        new_naming = [new if i == current else i for i in self.naming_order]

        return SegmentationSlice(new_data, new_naming)

    def add(self, segment: str, value: int) -> "SegmentationSlice":
        """Create a copy of slice with new segment added."""
        if segment in self._data:
            raise ValueError(f"{segment} segment already in slice")

        data = self._data | {segment: value}
        naming = self.naming_order + (segment,)
        return SegmentationSlice(data, naming)

    def remove(self, segment: str) -> "SegmentationSlice":
        """Create a copy of slice with segment removed."""
        if segment not in self._data:
            raise ValueError(f"{segment} segment not in slice")

        data = self._data.copy()
        data.pop(segment)
        naming = [i for i in self.naming_order if i != segment]
        return SegmentationSlice(data, naming)


class SegmentationInput(BaseConfig):
    """
    Input class for segmentation objects.

    Parameters
    ----------
    enum_segments: list[SegmentsSuper]
        Provide as strings matching enumerations in SegmentsSuper. In 99% of
        cases segments should be provided in this form. If a segment doesn't
        exist in the enumeration it should usually be added.
    subsets: dict[str, list[int]]
        Define any segments you want subsets of. Keys should match values in
        'enum_segments', values should be lists defining values from the
        segment which should be included.
    custom_segments: list[Segment]
        User defined segments which don't appear in SegmentsSuper. As stated,
        in almost all cases a missing segment should be added to that class,
        and this option is a contingency.
    naming_order: list[str]
        The naming order of the segments. This primarily affects the index
        order of the multi-index formed from the segmentation.
    """

    naming_order: list[str]
    enum_segments: list[SegmentsSuper] = pydantic.Field(default_factory=list)
    custom_segments: list[Segment] = pydantic.Field(default_factory=list)
    subsets: dict[str, list[int]] = pydantic.Field(default_factory=dict)

    # pylint doesn't seem to understand pydantic default fields
    # pylint: disable=not-an-iterable,no-member
    @pydantic.model_validator(mode="after")
    def no_copied_names(self):
        """Validate the custom_segments do not clash with existing segments."""
        for seg in self.custom_segments:
            if seg.name in SegmentsSuper.values():
                raise ValueError(
                    "There is already a segment defined with name "
                    f"{seg.name}. Segment names must be unique "
                    "even if the existing segment isn't in this "
                    "segmentation. This error is raised on the "
                    "first occurrence so it is possible there is "
                    "more than one clash. 'caf.base.SegmentsSuper.values' "
                    "will list all existing segment names."
                )
        return self

    @pydantic.model_validator(mode="after")
    def names_match_segments(self):
        """Validate that naming order names match segment names."""
        v = self.naming_order
        seg_names = [i.value for i in self.enum_segments]
        if len(self.custom_segments) > 0:
            seg_names += [i.name for i in self.custom_segments]

        if set(seg_names) != set(v):
            raise ValueError("Names provided for naming_order do not match names in segments")

        return self

    @pydantic.model_validator(mode="after")
    def enums(self):
        """Validate the subsets match segments."""
        for seg in self.subsets.keys():
            if seg not in [i.value for i in self.enum_segments]:
                raise ValueError(
                    f"{seg} is not a valid segment  " ", and so can't be a subset value."
                )
        return self

    # pylint: enable=not-an-iterable,no-member


class Segmentation:
    """
    Segmentation class for handling segmentation objects.

    Parameters
    ----------
        config: SegmentationInput
        Instance of SegmentationInput. See that class for details.
    """

    # pylint: disable=too-many-public-methods
    # Maybe there is a requirement to look into splitting functionality up
    # to make this class simpler but for now everything in here is useful

    # This currently isn't used and doesn't mean anything. In few places code
    # relating to time periods or time formats is included from normits_core but
    # never used.
    _time_period_segment_name = "tp"

    def __init__(self, config: SegmentationInput):
        self.input = config
        # unpack enum segments, applying subsets if necessary
        if config.subsets is None:
            enum_segments = [
                SegmentsSuper(string).get_segment() for string in config.enum_segments
            ]

        else:
            enum_segments = []
            for seg in config.enum_segments:
                segment = SegmentsSuper(seg).get_segment(subset=config.subsets.get(seg.value))
                enum_segments.append(segment)

        self.segments = config.custom_segments + enum_segments
        self.naming_order = config.naming_order

    @property
    def seg_dict(self):
        """Access segments in dict form."""
        return {seg.name: seg for seg in self.segments}

    def get_segment(self, seg_name: str) -> Segment:
        """Get a segment based on its name."""
        return self.seg_dict[seg_name]

    def __iter__(self):
        """Iterate through seg_dict."""
        return self.seg_dict.__iter__()

    @property
    def names(self):
        """Return the names of all segments."""
        return [seg.name for seg in self.segments]

    @property
    def seg_descriptions(self):
        """Return a list of segment descriptions."""
        return [seg.values.values() for seg in self.segments]

    @property
    def seg_vals(self):
        """Return all segmentation values."""
        return [seg.values.keys() for seg in self.segments]

    def get_segment_values(self, segment: str) -> list[int]:
        """Get list of values for segment, checks for subsets."""
        if segment in self.input.subsets:
            return self.input.subsets[segment].copy()
        if segment in self.seg_dict:
            return self.seg_dict[segment].int_values
        raise KeyError(f"{segment} segment not found in segmentation")

    def lookup_ind(self):
        """Produce an index from the lookups of the segments."""
        lookups = []
        no_prod = []
        copy_iterator = self.segments.copy()
        for own_seg in self.segments:
            for other_seg in copy_iterator:
                if other_seg.name == own_seg.name:
                    continue
                if other_seg.name in own_seg.lookup_segs:
                    lookup = (
                        pd.DataFrame(index=own_seg.lookup_indices(other_seg.name))
                        .reset_index()
                        .rename(columns={"dummy": own_seg.name})
                    )
                    lookup = lookup[
                        (lookup[own_seg.name].isin(own_seg.int_values))
                        & (lookup[other_seg.name].isin(other_seg.int_values))
                    ]
                    lookup.set_index([own_seg.name, other_seg.name], inplace=True)
                    # lookup['val'] = 1
                    lookups.append(lookup)
                    no_prod.append(own_seg.name)
                    no_prod.append(other_seg.name)
        joined = None
        if len(lookups) > 0:
            joined = lookups[0]
            for lookup in lookups[1:]:
                if len(set(lookup.index.names).intersection(joined.index.names)) > 0:
                    joined = joined.join(lookup, how="inner")
                else:
                    new_ind = product_multiindex(joined.index, lookup.index)
                    joined = pd.DataFrame(index=new_ind)
        no_prod = list(set(no_prod))
        return joined, no_prod

    def ind(self):
        """
        Return a pandas MultiIndex of the segmentation.

        This is by default just a product of all segments given, taking
        exclusions into account if any exist between segments.
        """
        joined, no_prod = self.lookup_ind()
        prod = [self.seg_dict[i].int_values for i in self.naming_order if i not in no_prod]
        names = [i for i in self.naming_order if i not in no_prod]
        if len(prod) == 0:
            return joined.reorder_levels(self.naming_order).sort_index().index
        index = pd.MultiIndex.from_product(prod, names=names)
        if joined is not None:
            index = product_multiindex(index, joined.index)
        df = pd.DataFrame(index=index).reorder_levels(self.naming_order).reset_index()
        copy_iterator = self.naming_order.copy()
        for own_seg in self.segments:
            for other_seg in copy_iterator:
                if other_seg == own_seg.name:
                    continue
                if other_seg in own_seg.exclusion_segs:
                    dropper = own_seg.drop_indices(other_seg)
                    df = df.set_index([own_seg.name, other_seg])
                    mask = ~df.index.isin(dropper)
                    df = df[mask].reset_index()
                # pylint: enable=protected-access
        return df.set_index(self.naming_order).sort_index().index

    def has_time_period_segments(self) -> bool:
        """Check whether this segmentation has time period segmentation.

        Returns
        -------
        has_time_period_segments:
            True if there is a time_period segment in this segmentation,
            False otherwise
        """
        return self._time_period_segment_name in self.naming_order

    # pylint: disable=too-many-branches
    @classmethod
    def validate_segmentation(
        cls,
        source: Union[Path, pd.DataFrame],
        segmentation: Segmentation,
        escalate_warning: bool = False,
        cut_read: bool = False,
    ) -> tuple[Segmentation, bool]:
        """
        Validate a segmentation from either a path to a csv, or a dataframe.

        This could either be purely a segmentation, or data with a segmentation
        index.

        Parameters
        ----------
        source : Path | pd.DataFrame
            Either a path to a csv containing a segmentation or a dataframe
            containing a segmentation. If source is a dataframe the
            segmentation should not form the index.
        segmentation : Segmentation
            The segmentation you expect 'source' to match.
        escalate_warning: bool = False
            Whether to escalate warnings to errors.
        cut_read: bool = False
            Whether to cut the read in Segmentation to the expected one.

        Returns
        -------
        Segmentation class
        """
        if escalate_warning:
            warnings.filterwarnings("error", category=SegmentationWarning)
        if isinstance(source, Path):
            df = pd.read_csv(source)
        else:
            df = source
        naming_order = segmentation.naming_order
        conf = copy.deepcopy(segmentation.input)
        if df.index.names == naming_order:
            df.sort_index(inplace=True)
            read_index = df.index
        else:
            # Try to build index from df columns
            try:
                df = df.set_index(naming_order).sort_index()
            # Assume the index is already correct but reorder to naming_order
            except (KeyError, AttributeError):
                df = df.reorder_levels(naming_order).sort_index()
            read_index = df.index
        # Index to validate against
        built_index = segmentation.ind()
        # I think an error would already be raised at this point
        if built_index.names != read_index.names:
            raise ValueError(
                "The read in segmentation does not match the given parameters. "
                "The segment names are not correct."
            )

        # Perfect match, return segmentation with no more checks
        if read_index.equals(built_index):
            return segmentation, False
        if not cut_read:
            if len(read_index) > len(built_index):
                raise IndexError(
                    "The segmentation of the read in dvector data "
                    "does not match the expected segmentation. This "
                    "is likely due to unconsidered exclusions."
                    f"{read_index.difference(built_index)} in read in index but not "
                    f"in expected index."
                )
        for name in built_index.names:
            built_level = set(built_index.get_level_values(name))
            read_level = set(read_index.get_level_values(name))
            # This level matches, check the next one
            if read_level == built_level:
                continue
            # The input segmentation should have had subsets defined. warn user but allow
            if read_level.issubset(built_level):
                warnings.warn(
                    f"Read in level {name} is a subset of the segment. If this was not"
                    f" expected check the input segmentation.",
                    SegmentationWarning,
                    stacklevel=2,
                )
                # Define the read subset in the generated config
                if name in [seg.value for seg in conf.enum_segments]:
                    if len(conf.subsets) > 0:
                        conf.subsets.update({name: list(read_level)})
                    else:
                        conf.subsets = {name: list(read_level)}
                else:
                    raise SegmentationError(f"{name} segment does not match " f"the data.")
            # Not a subset so doesn't match completely
            else:
                raise ValueError(
                    f"The segment for {name} does not match the inbuilt definition."
                    f"Check for mistakes in the read in segmentation, or redefine the"
                    f"segment with a different name."
                )

        built_segmentation = cls(conf)
        built_index = built_segmentation.ind()
        if read_index.equals(built_index):
            return built_segmentation, False
        # Still doesn't match, this is probably an exclusion error. User should check that
        # proper exclusions are defined in SegmentsSuper.
        if built_index.equals(built_index.intersection(read_index)):
            if cut_read:
                return built_segmentation, False
            raise SegmentationError(
                "Read data contains rows not in the generated segmentation. "
                "If you want this data to simply be cut to match, set 'cut_read=True'"
            )
        if read_index.equals(built_index.intersection(read_index)):
            warnings.warn(
                "Combinations missing from the read in data. This may mean an exclusion should "
                "be defined but isn't. The data will be expanded to the expected segmenation, and "
                "infilled with zeroes."
            )
            return built_segmentation, True
        raise ValueError(
            "The read in segmentation does not match the given parameters. The segment names"
            " are correct, but segment values don't match. This could be due to an incompatibility"
            " between segments which isn't reflected in the loaded in the segmentation, or it could be"
            " an out of date in built segmentation in the caf.base package. The first place to "
            "look is the SegmentsSuper class."
        )

    # pylint: enable=too-many-branches

    def translate_segment(
        self,
        from_seg: str | Segment,
        to_seg: str | Segment,
        reverse: bool = False,
        drop_from: bool = True,
    ) -> tuple[Segmentation, pd.Series]:
        """
        Translate one of the segments making up the segmentation.

        Parameters
        ----------
        from_seg: str | Segment
            The segment to translate. Must be contained within this segmentation.
        to_seg: str | Segment
            The segment to translate to. A translation must be defined for these two segments.
        reverse: bool = False
            Whether to perform this translation in reverse (i.e. from more aggregate to less).
            This will still be performed without factoring so is only appropriate for intensive
            properties like growth factors, not extensive properties like travel demand.
        drop_from: bool = True
            Whether to drop the old segment from the resulting segmentation.
        """
        if isinstance(to_seg, str):
            if to_seg in SegmentsSuper.values():
                to_seg = SegmentsSuper(to_seg).get_segment()
        if isinstance(from_seg, str):
            from_seg = self.get_segment(from_seg)
        if from_seg not in self.segments:
            raise SegmentationError(
                "The segment being translated from is not in the current segmentation."
            )
        to_seg, lookup = from_seg.translate_segment(to_seg, reverse=reverse)
        new_conf = self.input.model_copy(deep=True)
        if drop_from:
            if SegmentsSuper(from_seg.name) in new_conf.enum_segments:
                new_conf.enum_segments.remove(SegmentsSuper(from_seg.name))
            else:
                new_conf.custom_segments.remove(from_seg)
            new_conf.naming_order[new_conf.naming_order.index(from_seg.name)] = to_seg.name
        else:
            new_conf.naming_order.append(to_seg.name)
        try:
            new_conf.enum_segments.append(SegmentsSuper(to_seg.name))
        except ValueError:
            new_conf.custom_segments.append(to_seg)
        return Segmentation(new_conf), lookup

    def reinit(self):
        """Regenerate Segmentation from its input."""
        return Segmentation(self.input)

    def save(self, out_path: PathLike, mode: Literal["hdf", "yaml"] = "hdf"):
        """
        Save a segmentation to either a yaml file or an hdf file if part of a DVector.

        Parameters
        ----------
        out_path: PathLike
            Path to where the data should be saved. The file extension must
            match 'mode'
        mode: Literal["hdf", "yaml"]
            Currently only can be 'hdf' or 'yaml'. How to save the file.
        """
        if mode == "hdf":
            with h5py.File(out_path, "a") as h_file:
                h_file.create_dataset(
                    "segmentation", data=self.input.to_yaml().encode("utf-8")
                )

        elif mode == "yaml":
            self.input.save_yaml(Path(out_path))

        else:
            raise ValueError(f"Mode must be either 'hdf' or 'yaml', not {mode}")

    @classmethod
    def load(cls, in_path: PathLike, mode: Literal["hdf", "yaml"] = "hdf") -> Segmentation:
        """
        Load the segmentation from a file, either an hdf or csv file.

        Parameters
        ----------
        in_path: PathLike
            Path to the file. File extension must match 'mode'
        mode: Literal["hdf", "yaml"], default "hdf"
            Mode to load in, either 'hdf' or 'yaml'

        Returns
        -------
        Segmentation class
        """
        # pylint: disable=no-member
        if mode == "hdf":
            with h5py.File(in_path, "r") as h_file:
                yam_load = h_file["segmentation"][()].decode("utf-8")
                config = SegmentationInput.from_yaml(yam_load)
        # pylint: enable=no-member

        elif mode == "yaml":
            config = SegmentationInput.load_yaml(Path(in_path))

        else:
            raise ValueError(f"Mode must be either 'hdf' or 'yaml', not {mode}")

        return cls(config)

    def __copy__(self):
        """Return a copy of this class."""
        return self.copy()

    def __eq__(self, other) -> bool:
        """Override the default implementation."""
        if not isinstance(other, Segmentation):
            return False

        if self.naming_order != other.naming_order:
            return False

        if set(self.names) != set(other.names):
            return False

        return True

    def __len__(self):
        """Return length of segmentation."""
        return len(self.ind())

    def __add__(self, other):
        """
        Combine two segmentations without duplicates.

        Order of naming_order in resulting segmentation will have self before
        other. This name may be misleading as this is the method used for most
        of the dunder methods in DVector for combining resulting segmentations.
        """
        enum_in = set(self.input.enum_segments + other.input.enum_segments)
        cust_in = self.input.custom_segments
        for seg in other.input.custom_segments:
            if seg.name not in [i.name for i in cust_in]:
                cust_in.append(seg)
        subsets = self.input.subsets.copy()
        for seg, vals in subsets.items():
            if seg in other.input.subsets.keys():
                subsets[seg] = list(set(vals).intersection(other.input.subsets[seg]))
        for seg, vals in other.input.subsets.items():
            if seg not in subsets:
                subsets[seg] = vals
        naming_order = ordered_set(self.naming_order, other.naming_order)
        config = SegmentationInput(
            enum_segments=enum_in,
            subsets=subsets,
            custom_segments=cust_in,
            naming_order=naming_order,
        )
        return Segmentation(config)

    def overlap(self, other: Segmentation | list[str]):
        """Check the overlap in segments between two segmentations."""
        if isinstance(other, Segmentation):
            return set(self.names).intersection(other.names)
        return set(self.names).intersection(other)

    def subset_difference(self, other: Segmentation):
        """
        Return the difference between self and other's subsets.

        Parameters
        ----------
        other: Segmentation
            Segmentation to compare to self.

        Returns
        -------
        missing_self: dict[str, int]
            Subset values in other but not in self
        missing_other: dict[str, int]
            Subset values in self but not in other
        """
        if (self.input.subsets is None) & (other.input.subsets is None):
            return None
        self_subsets = [i for i in self.input.subsets.keys() if i in other.names]
        other_subsets = [j for j in other.input.subsets.keys() if j in self.names]
        if (len(self_subsets) == 0) & (len(other_subsets) == 0):
            return None
        missing_other = {}
        for name in other_subsets:
            full = SegmentsSuper(name).get_segment()
            if name not in self.input.subsets.keys():
                missing_other[name] = [
                    i for i in full.int_values if i not in other.input.subsets[name]
                ]
            elif self.input.subsets[name] != other.input.subsets[name]:
                missing_list = [
                    i for i in self.input.subsets[name] if i not in other.input.subsets[name]
                ]
                if len(missing_list) > 0:
                    missing_other[name] = missing_list
        missing_self = {}
        for name in self_subsets:
            full = SegmentsSuper(name).get_segment()
            if name not in other.input.subsets.keys():
                missing_self[name] = [
                    i for i in full.int_values if i not in self.input.subsets[name]
                ]
            elif self.input.subsets[name] != other.input.subsets[name]:
                missing_list = [
                    i for i in other.input.subsets[name] if i not in self.input.subsets[name]
                ]
                if len(missing_list) > 0:
                    missing_self[name] = missing_list
        return missing_self, missing_other

    def is_subset(self, other: Segmentation):
        """Check whether self is a subset of other."""
        return bool(self.overlap(other) == set(self.names))

    def __sub__(self, other):
        """
        Return segments in self but not in other.

        This method will not error if other contains segments not in self.
        """
        return [self.get_segment(i) for i in self.naming_order if i not in other.naming_order]

    def __ne__(self, other) -> bool:
        """Override the default implementation."""
        return not self.__eq__(other)

    def copy(self):
        """Copy an instance of this class."""
        return Segmentation(config=self.input.model_copy(deep=True))

    def aggregate(self, new_segs: list[str]):
        """
        Aggregate segmentation to a subset of the segmentation.

        This method isn't exactly an aggregation, it just removes segments.
        It is called aggregate as currently it is the segmentation component
        of the aggregate method in DVector.

        Parameters
        ----------
        new_segs: The new segmentation. All must be in the current segmentation. The order these are
        given in determine the naming order of the returned Segmentation.
        """
        custom = None
        subsets = None

        for i in new_segs:
            if i not in self.names:
                raise SegmentationError(
                    f"{i} is not in the current segmentation, "
                    "so cannot be aggregated to. This is the "
                    "first segment raising an error, and there "
                    "may be more."
                )

        if self.input.custom_segments is not None:
            custom = self.input.custom_segments.copy()
            for seg in self.input.custom_segments:
                if seg.name not in new_segs:
                    custom.remove(seg)

        enum_segs = self.input.enum_segments.copy()
        for enum_seg in self.input.enum_segments:
            if enum_seg.value not in new_segs:
                enum_segs.remove(enum_seg)

        if self.input.subsets is not None:
            subsets = dict()
            for key, val in self.input.subsets.items():
                if key in new_segs:
                    subsets.update({key: val})

        conf = SegmentationInput(
            enum_segments=enum_segs,
            subsets=subsets,
            custom_segments=custom,
            naming_order=new_segs,
        )
        return Segmentation(conf)

    def add_segment(
        self,
        new_seg: Segment | SegmentsSuper | str,
        subset: Optional[dict[str, list[int]]] = None,
        new_naming_order: Optional[list[str]] = None,
    ):
        """
        Add a new segment to a segmentation.

        Parameters
        ----------
        new_seg: Segment
            The new segment to be added. This will be checked and added as an
            enum_segment if it exists as such, and as a custom segment if not.
            This must be provided as a Segment type, and can't be a string to pass
            to the SegmentSuper enum class

        subset: Optional[dict[str, list[int]]] = None
            A subset definition if the new segmentation is a subset of an existing
            segmentation. This need only be provided for an enum_segment.

        new_naming_order: Optional[list[str]] = None
            The naming order of the resultant segmentation. If not provided,
            the new segment will be appended to the end.

        Returns
        -------
        Segmentation
        """
        out_segmentation = self.copy()
        custom = True
        if isinstance(new_seg, str):
            inner_seg: Segment = SegmentsSuper(new_seg).get_segment()
        elif isinstance(new_seg, SegmentsSuper):
            inner_seg = new_seg.get_segment()
        else:
            inner_seg = new_seg
        new_name = inner_seg.name
        if new_name in SegmentsSuper.values():
            custom = False

        if new_name in self.names:
            raise ValueError(f"{new_name} already contained in segmentation.")
        if custom:
            out_segmentation.input.custom_segments.append(inner_seg)
        else:
            out_segmentation.input.enum_segments.append(SegmentsSuper(new_name))
            if inner_seg.values != SegmentsSuper(new_name).get_segment().values:
                out_segmentation.input.subsets.update({new_name: inner_seg.int_values})
        if new_naming_order is not None:
            out_segmentation.input.naming_order = new_naming_order
        else:
            out_segmentation.input.naming_order.append(new_name)
        if subset is not None:
            out_segmentation.input.subsets.update(subset)
        return out_segmentation.reinit()

    def remove_segment(self, segment_name: str | Segment, inplace: bool = False):
        """
        Remove a segment from a segmentation.

        Parameters
        ----------
        segment_name: str
            The name of the segment to remove
        inplace: bool = False
            Whether to apply in place
        """
        if isinstance(segment_name, Segment):
            segment_name = segment_name.name
        if segment_name not in self.names:
            raise SegmentationError(
                f"{segment_name} is not in the current segmentation, so " f"cannot be removed."
            )
        if inplace:
            self.input.naming_order.remove(segment_name)
            if SegmentsSuper(segment_name) in self.input.enum_segments:
                self.input.enum_segments.remove(SegmentsSuper(segment_name))
            else:
                self.input.custom_segments.remove(self.get_segment(segment_name))
            if segment_name in self.input.subsets.keys():
                del self.input.subsets[segment_name]
            self.reinit()
            return self
        out_seg = self.input.model_copy(deep=True)
        out_seg.naming_order.remove(segment_name)
        if segment_name in SegmentsSuper.values():
            out_seg.enum_segments.remove(SegmentsSuper(segment_name))
        else:
            out_seg.custom_segments.remove(self.get_segment(segment_name))
        if segment_name in out_seg.subsets.keys():
            del out_seg.subsets[segment_name]
        return Segmentation(out_seg)

    def update_subsets(self, extension: dict[str, int | list[int]], remove=False):
        """
        Add to subsets dict.

        Parameters
        ----------
        extension: dict[str, int]
            Values to add to the subsets dict. This will still work if subsets
            is currently empty.
        """
        out_seg = self.input.copy()
        for key, val in extension.items():

            if key not in self.names:
                raise ValueError(
                    f"{key} not in current segmentation, so can't " "be added to subsets"
                )
            if isinstance(val, int):
                val = [val]
            if key in out_seg.subsets:
                if remove:
                    out_seg.subsets[key] = list(set(out_seg.subsets[key]) - set(val))
                else:
                    out_seg.subsets[key] = list(set(out_seg.subsets[key] + val))
            else:
                if remove:
                    out_seg.subsets[key] = list(set(self.seg_dict[key].values) - set(val))
                else:
                    out_seg.subsets.update({key: val})
        return Segmentation(out_seg)

    def generate_slice_name(self, slice_: SegmentationSlice) -> str:
        """Generate name for a slice of the segmentation from parameters.

        Parameters
        ----------
        slice_ : SegmentationSlice
            Parameters to generate name from.

        Returns
        -------
        str
            Name of slice from parameters in the form
            "{name}{value}_{name 2}{value 2}" e.g. "p1_m3", where name
            is the segment alias if it has one. The order of the
            parameters is defined by the segmentation `naming_order`.
        """
        return slice_.generate_name(self.seg_dict)

    def iter_slices(
        self, filter_: dict[str, int] | None = None
    ) -> Iterator[SegmentationSlice]:
        """Iterate through parameters for all segmentation slices.

        Parameters
        ----------
        filter_
            Optional parameters to filter slices e.g. {"p": 1}
            will exclude any slices where p != 1. If not given
            will return all slices.

        Yields
        ------
        SegmentationSlice
            Parameters for an individual slice.
        """
        slices: pd.DataFrame = self.ind().to_frame(index=False)

        if filter_ is not None:
            if not set(filter_) <= set(self.names):
                missing = set(filter_) - set(self.names)
                raise ValueError(
                    f"parameters given for segments not present in segmentation: {missing}"
                )

            mask = pd.Series(True, index=slices.index)
            for nm, value in filter_.items():
                mask = mask & (slices[nm] == value)

            slices = slices.loc[mask]

        if len(slices) == 0:
            warnings.warn(
                f"No slices found in segmentation with filter {filter_}", RuntimeWarning
            )

        params: NamedTuple
        for params in slices.itertuples(index=False):
            yield SegmentationSlice(params._asdict())

    def convert_slice_tuple(self, slice_tuple: tuple[int, ...]) -> SegmentationSlice:
        """Convert a segmentation slice tuple into SegmentationSlice.

        Parameters
        ----------
        slice_tuple : tuple[int, ...]
            Tuple containing segment values for a single slice.

        Returns
        -------
        SegmentationSlice
            Slice parameters.

        Raises
        ------
        ValueError
            If the tuple given doesn't have the expected number of values.

        See Also
        --------
        SegmentationSlice.from_tuple : Method to convert a tuple into a SegmentationSlice.
        """
        return SegmentationSlice.from_tuple(slice_tuple, self.naming_order)

    def convert_slice_name(self, name: str) -> SegmentationSlice:
        """Convert segmentation slice name into parameters.

        Parameters
        ----------
        name : str
            Segmentation slice name e.g. "p1_m3".

        Returns
        -------
        SegmentationSlice
            Parameters for the slice.

        Raises
        ------
        ValueError
            If multiple values are found for a single segment.
        KeyError
            If any segments are missing from the name.
        """
        missing = []
        params = {}

        for nm in self.naming_order:
            segment = self.get_segment(nm)
            matched = segment.extract_values(name)

            if len(matched) == 0:
                if segment.alias is not None:
                    missing.append(f"{nm} ({segment.alias})")
                else:
                    missing.append(nm)
                continue
            if len(matched) > 1:
                raise ValueError(f"found multiple values for {nm} segment in '{name}'")

            params[nm] = matched[0]

        if len(missing) > 0:
            raise KeyError(
                f"missing segments when generating params from name: {', '.join(missing)}"
            )

        return SegmentationSlice(params)

    def find_files(
        self, folder: Path, template: str, suffixes: collections.abc.Sequence[str]
    ) -> dict[SegmentationSlice, Path]:
        """Find files split by segmentation in given `folder`.

        Checks if files exist with the generated filename and if
        any are missing performs more in-depth search to find them.
        In-depth search finds all files in `folder` and checks
        if the filename contains all required segment parameters.

        Parameters
        ----------
        folder : Path
            Folder to search within, doesn't look in sub-folders.
        template : str
            Template for filenames, will be formatted with the slice
            name so should contain "{slice_name}". File extension shouldn't
            be included e.g "test" instead of "test.csv".
        suffixes : collections.abc.Sequence[str]
            File extensions (suffixes) to search for, will find the first
            matching file.

        Returns
        -------
        dict[SegmentationSlice, Path]
            Files found for each slice (value), with the
            slice parameters (key).

        Raises
        ------
        FileNotFoundError
            If a file cannot be found for all possible segments.
        FileExistsError
            If multiple files are found for the same slice when performing
            more in-depth search.

        Warns
        -----
        RuntimeWarning
            If any additional unexpected files are found during the
            in-depth search.
        """
        missing = {}
        filepaths = {}
        for params in self.iter_slices():
            name = self.generate_slice_name(params)
            filename = template.format(slice_name=name)

            try:
                path = ctk.io.find_file_with_name(folder, filename, suffixes)
            except FileNotFoundError:
                missing[params] = filename
                continue

            filepaths[params] = path

        if len(missing) == 0:
            return filepaths
        warnings.warn(
            f"Found {len(filepaths)} ({len(filepaths) / len(self):.0%}) files,"
            f" {len(missing)} missing, with initial search. Performing in-depth search.",
            RuntimeWarning,
        )

        # Find any other files in the folder and attempt to parse the name to see
        # if they match the missing slices
        for path in folder.iterdir():
            if not path.is_file():
                continue
            if "".join(path.suffixes) not in suffixes:
                continue
            if path in filepaths.values():
                continue

            try:
                slice_ = self.convert_slice_name(path.stem)
            except (ValueError, KeyError) as exc:
                warnings.warn(
                    f"Found unexpected file while searching: {path}\n{exc}", RuntimeWarning
                )
                continue

            if slice_ in filepaths:
                raise FileExistsError(
                    f"found multiple files for {slice_}: '{path.name}' and '{filepaths[slice_]}'"
                )

            if slice_ in missing:
                filepaths[slice_] = path
                missing.pop(slice_)
            else:
                warnings.warn(f"Found unexpected file while searching: {path}", RuntimeWarning)

        if len(missing) == 0:
            return filepaths

        missing_names = ", ".join(f"'{i}'" for i in missing.values())
        raise FileNotFoundError(
            f"missing {len(missing)} ({len(missing) / len(self):.0%})"
            f' files from "{folder}": {missing_names}'
        )

    def validate_slice(
        self, slice_: SegmentationSlice, fix_order: bool = False
    ) -> SegmentationSlice:
        """Validate slice contains expected segments, and no extra.

        Parameters
        ----------
        slice_
            Segmentation slice for validation.
        fix_order
            If True will return a copy of the slice_ with fixed naming
            order instead of raising an error, default is False.

        Raises
        ------
        ValueError
            If segments are missing, extra segments are found or
            naming order is different.
        """
        missing = list(filter(lambda x: x not in slice_, self.seg_dict))
        if len(missing) > 0:
            raise ValueError(
                f"{len(missing)} segments missing from slice: {', '.join(missing)}"
            )

        extra = list(filter(lambda x: x not in self.seg_dict, slice_.data))
        if len(extra) > 0:
            raise ValueError(
                f"{len(extra)} segments in slice but not segmentation: {', '.join(extra)}"
            )

        if slice_.naming_order != self.naming_order:
            if not fix_order:
                raise ValueError(
                    "slice naming order is incorrect got "
                    f"{slice_.naming_order} but expected {self.naming_order}"
                )
            return SegmentationSlice(slice_.data, self.naming_order.copy())

        return slice_


# # # FUNCTIONS # # #
def ordered_set(list_1: list, list_2: list) -> list:
    """Take in two lists and combine them, removing duplicates but preserving order."""
    combined_list = list_1 + list_2
    unique_list = []
    for item in combined_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def product_multiindex(multi_index1, multi_index2):
    """
    Take two MultiIndex objects and return their Cartesian product as a new MultiIndex.

    Parameters
    ----------
    multi_index1 : pd.MultiIndex
        The first MultiIndex.
    multi_index2: pd.MultiIndex
        The second MultiIndex.

    Returns
    -------
    pd.MultiIndex: A new MultiIndex which is the Cartesian product of the two input MultiIndices.
    """
    # Generate Cartesian product
    product = list(itertools.product(multi_index1, multi_index2))

    # Combine tuples from both MultiIndices
    combined_tuples = [tuple(list(x[0]) + list(x[1])) for x in product]

    # Combine names from both MultiIndices
    combined_names = list(multi_index1.names) + list(multi_index2.names)

    # Create a new MultiIndex from combined tuples and names
    combined_index = pd.MultiIndex.from_tuples(combined_tuples, names=combined_names)

    return combined_index
