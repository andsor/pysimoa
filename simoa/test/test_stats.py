# -*- coding: utf-8 -*-


import functools
import itertools

import hypothesis
import hypothesis.strategies as st
import numpy as np
import simoa.stats


@hypothesis.given(
    st.lists(
        elements=st.one_of(
            st.integers(min_value=10, max_value=1000000),
            st.floats(min_value=1e1, max_value=1e6),
        ),
        min_size=2,
    ),
)
def test_online_variance_incremental(data):
    result = functools.reduce(
        simoa.stats.online_variance,
        ((1, data_point, 0.0) for data_point in data),
    )
    assert result[0] == len(data)
    np.testing.assert_allclose(result[1], np.mean(data))
    np.testing.assert_allclose(
        result[2],
        np.sum(np.square(np.subtract(data, np.mean(data)))),
    )


@hypothesis.given(
    st.lists(
        elements=st.lists(
            elements=st.one_of(
                st.integers(min_value=10, max_value=1000000),
                st.floats(min_value=1e1, max_value=1e6),
            ),
            min_size=2,
        ),
        min_size=2,
    ),
)
def test_online_variance_several(data):
    reduced = [
        functools.reduce(
            simoa.stats.online_variance,
            ((1, data_point, 0.0) for data_point in samples),
        )
        for samples in data
    ]
    result = functools.reduce(simoa.stats.online_variance, reduced)
    alldata = list(itertools.chain(*data))
    assert result[0] == len(alldata)
    np.testing.assert_allclose(result[1], np.mean(alldata))
    np.testing.assert_allclose(
        result[2],
        np.sum(np.square(np.subtract(alldata, np.mean(alldata)))),
    )


def test_von_neumann_ratio_test_invocation():
    simoa.stats.von_neumann_ratio_test(
        data=np.ones(1024),
        alpha=0.32
    )


def test_von_neumann_ratio_test_pass():
    np.random.seed(1)
    assert simoa.stats.von_neumann_ratio_test(
        data=np.random.rand(1024),
        alpha=0.2,
    )


def test_von_neumann_ratio_test_reject():
    np.random.seed(4)
    assert not simoa.stats.von_neumann_ratio_test(
        data=np.random.rand(1024),
        alpha=0.2,
    )
