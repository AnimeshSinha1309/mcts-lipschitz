import pytest

import montecomb


def test_basic_properties():
    df = montecomb.dataset.subset_additions.SubsetAdditionDataset(10)
    assert df.n == 10


@pytest.mark.xfail
def test_evaluations():
    df = montecomb.dataset.subset_additions.SubsetAdditionDataset(10)
    assert abs(df(1023) - df(511)) <= 2.1
