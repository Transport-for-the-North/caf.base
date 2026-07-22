"""Core classes and definitions used in the Common Analytical Framework."""

# ruff noqa:F401
from caf.base import (
	data_structures as data_structures,
	segmentation as segmentation,
	segments as segments,
	zoning as zoning,
)
from caf.base.data_structures import DVector as DVector
from caf.base.segmentation import (
	Segmentation as Segmentation,
	SegmentationInput as SegmentationInput,
)
from caf.base.segments import Segment as Segment
from caf.base.zoning import (
	BalancingZones as BalancingZones,
	ZoningSystem as ZoningSystem,
)

from ._version import __version__ as __version__
