import pytest

import montecomb


def test_basic_properties():
    df = montecomb.dataset.subset_additions.SubsetAdditionDataset(10)
    assert df.n == 10


def test_evaluations():
    df = montecomb.dataset.subset_additions.SubsetAdditionDataset(10)
    assert abs(df(8) - df(0)) <= 2
