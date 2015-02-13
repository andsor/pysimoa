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


import logging
import math

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
    assert env['n'] == env['m'] * env['k']


def test_nskart_step_1_initial_batch_size_skewed_data():
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 10
    assert env['n'] == env['m'] * env['k']


def test_nskart_step_1_minimum_initial_batch_size():
    data = np.random.geometric(p=0.99, size=128000)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 16
    assert env['n'] == env['m'] * env['k']


def test_nskart_step_1_initial_nonspaced_batch_means():
    data = np.ones(1280)
    env = simoa.nskart._step_1(data)
    numpy.testing.assert_allclose(env['Y_j(m)'], np.ones(1280))
    assert env['Y_j(m)'].size == env['k']


def test_nskart_step_1_initial_nonspaced_batch_means_skewed_data():
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    assert env['m'] == 10
    assert data[10:20].mean() == env['Y_j(m)'][1]
    assert env['Y_j(m)'].size == env['k']


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


def test_nskart_step3a_pass_randomness_test():
    np.random.seed(1)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)
    assert env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    assert env["k'"] == env['k']


def test_nskart_step3a_fail_randomness_test():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]


def test_nskart_step3bd():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    print(env)
    assert env['d'] == 1
    assert env["k'"] == env['k'] / 2
    assert env['Y_j(m,d)'].size == env["k'"]
    assert env['Y_j(m,d)'][0] == env['Y_j(m)'][1]
    assert env['Y_j(m,d)'][1] == env['Y_j(m)'][3]
    assert env['Y_j(m,d)'][-1] == env['Y_j(m)'][-1]


def test_nskart_iterate_step3bd():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3bd(env)
    print(env)
    assert env['d'] == 2
    assert env["k'"] == math.floor(env['k'] / 3) == 426
    assert env['k'] % 3 == 2
    assert env['Y_j(m,d)'].size == env["k'"]
    assert env['Y_j(m,d)'][0] == env['Y_j(m)'][2]
    assert env['Y_j(m,d)'][1] == env['Y_j(m)'][5]
    assert env['Y_j(m,d)'][-1] == env['Y_j(m)'][-1 - env['k'] % 3]


def test_nskart_step3c_pass():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3c(env)
    print(env)
    assert env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]


def test_nskart_step3c_fail():
    np.random.seed(22)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3c(env)
    print(env)
    assert not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]


@pytest.fixture
def nskart_step4_env_insufficient_data():
    np.random.seed(4960)
    data = np.random.geometric(p=0.99, size=1280)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)  # d == 1
    assert env['d'] == 1
    env = simoa.nskart._step_3c(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)  # d == 2
    assert env['d'] == 2
    env = simoa.nskart._step_3c(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)  # d == 3
    env = simoa.nskart._step_3c(env)  # fail randomness test
    assert (
        not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
        and not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    )
    return env


def test_nskart_step4_raises_insufficient_data(
    nskart_step4_env_insufficient_data
):
    with pytest.raises(simoa.nskart.NSkartInsufficientDataError):
        simoa.nskart._step_4(nskart_step4_env_insufficient_data)


def test_nskart_step4_continue_insufficient_data(
    caplog, nskart_step4_env_insufficient_data
):
    with caplog.atLevel(logging.ERROR, logger='simoa.nskart'):
        env = simoa.nskart._step_4(
            env=nskart_step4_env_insufficient_data,
            continue_insufficient_data=True
        )
    assert list(caplog.records())[-1].levelno == logging.ERROR
    assert env[simoa.nskart.NSKART_INSUFFICIENT_DATA_KEY]


@pytest.fixture
def nskart_step4_env_sufficient_data():
    np.random.seed(435)
    data = np.random.geometric(p=0.99, size=128000)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)  # d == 1
    assert env['d'] == 1
    env = simoa.nskart._step_3c(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)  # d == 2
    assert env['d'] == 2
    env = simoa.nskart._step_3c(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)  # d == 3
    env = simoa.nskart._step_3c(env)  # fail randomness test
    assert (
        not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
        and not env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    )
    return env


def test_nskart_step4_sufficient_data(nskart_step4_env_sufficient_data):
    env = simoa.nskart._step_4(nskart_step4_env_sufficient_data)
    assert env['b'] == 1
    assert env['k'] == 1152
    assert env['m'] == 23
    assert env['d'] == 0
    assert env['d^*'] == 10
    assert env['Y_j(m)'].size == env['k']


def test_nskart_step5a():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3c(env)
    assert env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_5a(env)

    assert env["k'"] == 904
    assert env['m'] == 14
    assert env['w'] == 144


def test_nskart_step5b():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3c(env)
    assert env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_5a(env)
    env = simoa.nskart._step_5b(env)
    assert env['Y_j(m)'].size == env["k'"]
    assert env['Y_j(m)'][1] == env['X_i'][158:172].mean()
    assert env['Y_j(m)'][-1] == env['X_i'][-14:].mean()


def test_nskart_step6():
    np.random.seed(7)
    data = np.random.geometric(p=0.99, size=12800)
    env = simoa.nskart._step_1(data)
    env = simoa.nskart._step_2(env)
    env = simoa.nskart._step_3a(env)  # fail randomness test
    assert not env[simoa.nskart.NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_3bd(env)
    env = simoa.nskart._step_3c(env)
    assert env[simoa.nskart.NSKART_SPACED_RANDOMNESS_TEST_KEY]
    env = simoa.nskart._step_5a(env)
    env = simoa.nskart._step_5b(env)
    env = simoa.nskart._step_6(env)
    assert env[simoa.nskart.NSKART_GRAND_AVERAGE_KEY] == (
        env['X_i'][env['w']:].mean()
    )


def test_nskart_invocation():
    return
    simoa.nskart(
        data=np.ones(12800),
        confidence_level=0.68,
    )
