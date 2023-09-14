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
import pytest
import pandas as pd
import numpy as np
from caf.core import segmentation
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #
@pytest.fixture(scope='session', name='multi-index')
def fix_mult():
    # Define the index levels
    level_a = ['A', 'B', 'C', 'D', 'E', 'F']
    level_b = ['G', 'H', 'I', 'J', 'K', 'L']
    level_c = ['M', 'N', 'O', 'P', 'Q', 'R']
    level_d = ['S', 'T', 'U', 'V', 'W', 'X']

    # Create a MultiIndex
    index = pd.MultiIndex.from_tuples(
        [(a, b, c, d) for a, b, c, d in zip(level_a, level_b, level_c, level_d)],
        names=['a', 'b', 'c', 'd'])

    # Create a DataFrame with random data
    data = np.random.rand(6, 1)

    df = pd.DataFrame(data, index=index, columns=['RandomData'])

@pytest.fixture(scope='session', name='expected_excl_ind')
def fix_excl_ind():

    return pd.MultiIndex.from_tuples([(1, 1),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
                (4, 1),
                (4, 2),
                (4, 3)],
               names=['test seg 1', 'test seg 2'])

class TestSegments:
    def test_naming_order(self, basic_segmentation, expected_excl_ind):
        names = basic_segmentation.ind.names
        assert names==['test seg 2', 'test seg 1']

    @pytest.mark.parametrize("segmentation",
                             ["excl_segmentation", "excl_segmentation_rev"])
    def test_exclusions(self, segmentation, expected_excl_ind, request):
        seg = request.getfixturevalue(segmentation)
        print(seg.ind)
        assert seg.ind.equals(expected_excl_ind)

# # # FUNCTIONS # # #
