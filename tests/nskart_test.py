# -*- coding: utf-8 -*-

'''

   Copyright 2015 The pysimoa Developers

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

'''


import numpy as np
import numpy.testing
import pytest
import simoa
import simoa.nskart


def test_nskart_step_1_raises_if_too_few_values():
    with pytest.raises(simoa.NSkartTooFewValues):
        simoa.nskart._step_1(np.ones(1279))


def test_nskart_step_1_initial_values():
    data = np.ones(1280)
    env = simoa.nskart._step_1(data)
    assert env['X_i'] is data
    assert env['N'] == data.size
    assert 'm' in env
    assert env['d'] == simoa.nskart.NSKART_INITIAL_NUMBER_OF_BATCHES_IN_SPACER
    assert (
        env['d^*'] == simoa.nskart.NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER
    )
    assert env['k'] == simoa.nskart.NSKART_INITIAL_BATCH_NUMBER
    assert env[simoa.nskart.NSKART_RANDOMNESS_TEST_SIGNIFICANCE_KEY] == (
        simoa.nskart.NSKART_RANDOMNESS_TEST_SIGNIFICANCE
    )
    assert env['b'] == 0


def test_nskart_step_1_initial_batch_size_unskewed_data():
    data = np.random.rand(1280)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 1


def test_nskart_step_1_initial_batch_size_skewed_data():
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 10


def test_nskart_step_1_minimum_initial_batch_size():
    data = np.random.geometric(p=0.99, size=128000)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 16


def test_nskart_step_1_initial_nonspaced_batch_means():
    data = np.ones(1280)
    env = simoa.nskart._step_1(data)
    numpy.testing.assert_allclose(env['Y_j'], np.ones(1280))
    assert env['Y_j'].size == env['k']


def test_nskart_step_1_initial_nonspaced_batch_means_skewed_data():
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 10
    assert data[10:20].mean() == env['Y_j'][1]
    assert env['Y_j'].size == env['k']


def test_nskart_step_2_nonskewed():
    data = np.random.rand(12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    assert env['d^*'] == (
        simoa.nskart.NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER
    )


def test_nskart_step_2_skewed():
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    assert env['d^*'] == (
        simoa.nskart.NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER_SKEWED
    )


def test_nskart_invocation():
    return
    simoa.nskart(
        data=np.ones(12800),
        confidence_level=0.68,
    )
