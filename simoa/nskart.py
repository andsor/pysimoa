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

from __future__ import division

import logging
import math
from collections import namedtuple

import numpy as np
import scipy.stats
from simoa.stats import von_neumann_ratio_test

logger = logging.getLogger(__name__)


# check python 3
if not (3 / 2 == 1.5):
    raise RuntimeError


ONE_SIGMA = scipy.stats.norm().cdf(1) - scipy.stats.norm().cdf(-1)


""" Minimum number of data points """
NSKART_MIN_DATA_POINTS = 1280

NSKART_INITIAL_NUMBER_OF_BATCHES_IN_SPACER = 0
NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER = 10
NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER_SKEWED = 3
NSKART_INITIAL_BATCH_NUMBER = 1280
NSKART_RANDOMNESS_TEST_SIGNIFICANCE = 0.20
NSKART_RANDOMNESS_TEST_SIGNIFICANCE_KEY = r'\alpha_{\text{ran}}'
NSKART_NONSPACED_RANDOMNESS_TEST_KEY = (
    'nonspaced batch means randomness test passed'
)
NSKART_SPACED_RANDOMNESS_TEST_KEY = (
    'spaced batch means randomness test passed'
)
NSKART_INSUFFICIENT_DATA_KEY = "insufficient data"
NSKART_GRAND_AVERAGE_KEY = r"\bar{Y}(m,k')"
NSKART_SAMPLE_VAR_KEY = r"S^2_{m,k'}"
NSKART_SAMPLE_LAG1_CORR = r"\hat{\varphi}_{Y(m)}"
NSKART_BATCHED_GRAND_MEAN_KEY = r"\bar{Y}(m,k'',d')"
NSKART_BATCHED_SAMPLE_VAR_KEY = r"S^2_{m,k'',d'}"
NSKART_BATCHED_SKEW_KEY = r"\hat{\mathcal{B}}_{m,k''}"


class NSkartException(BaseException):
    pass


class NSkartInsufficientDataError(NSkartException, RuntimeError):
    pass


class NSkartTooFewValues(NSkartException, ValueError):
    pass


NSkartReturnValue = namedtuple(
    'NSkartReturnValue',
    ['mean', 'ci', 'env']
)


def compute_nskart_interval(
    data, confidence_level=ONE_SIGMA, continue_insufficient_data=False
):
    """
    Compute the confidence interval for the steady-state mean

    Implements the N-Skart algorithm, "a nonsequential procedure designed to
    deliver a confidence interval for the steady-state mean of a simulation
    output process when the user supplies a single simulation-generated time
    series of arbitrary size and specifies the required coverage probability
    for a confidence interval based on that data set."

    Notes
    -----
    "N-Skart is a variant of the method of batch means that exploits separate
    adjustments to the half-length of the CI so as to account for the effects
    on the distribution of the underlying Student’s t-statistic that arise from
    skewness (nonnormality) and autocorrelation of the batch means.
    If the sample size is sufficiently large, then N-Skart delivers not only
    a CI but also a point estimator for the steady-state mean that
    is approximately free of initialization bias."

    References
    ----------

    .. [1] Tafazzoli, A.; Steiger, N.M.; Wilson, J.R.,
       "N-Skart: A Nonsequential Skewness- and Autoregression-Adjusted
       Batch-Means Procedure for Simulation Analysis,"
       Automatic Control, IEEE Transactions on , vol.56, no.2, pp.254,264, 2011
       doi: 10.1109/TAC.2010.2052137

"""

    # STEPS 1 -- 4
    env = _get_independent_data(data, continue_insufficient_data)

    # STEP 5a
    env = _step_5a(env)

    # STEP 5b
    env = _step_5b(env)

    # STEP 6
    env = _step_6(env)

    # STEP 7
    env = _step_7(env, confidence_level)

    return NSkartReturnValue(
        mean=env[NSKART_GRAND_AVERAGE_KEY],
        ci=env['CI'],
        env=env,
    )


def _get_independent_data(data, continue_insufficient_data=False):

    # STEP 1
    env = _step_1(data)

    while True:
        # STEP 2
        env = _step_2(env)

        # STEP 3

        # STEP 3a
        env = _step_3a(env)

        if env[NSKART_NONSPACED_RANDOMNESS_TEST_KEY]:
            # randomness test passed (failed-to-reject)
            return env

        # randomness test failed (independence hypothesis rejected)

        while env["d"] < env["d^*"]:
            # STEP 3b/d
            env = _step_3bd(env)

            # STEP 3c
            env = _step_3c(env)

            if env[NSKART_SPACED_RANDOMNESS_TEST_KEY]:
                # randomness test passed (failed-to-reject)
                return env

            # Continue with Step 3b/d
            continue

        # STEP 4
        env = _step_4(env, continue_insufficient_data)

        if NSKART_INSUFFICIENT_DATA_KEY in env:
            return env

        # continue with Step 2
        continue

    raise RuntimeError


def _compute_nonspaced_batch_means(env):
    return (
        env['X_i']
        [:env['m'] * env['k']]
        .reshape((env['k'], env['m']))
        .mean(axis=1)
    )


def _step_1(data):
    """
    Perform step 1 of the N-Skart algorithm

    Employs the `logging` module for output.

    Parameters
    ----------
    data: array-like
        Simulation output series
    """

    logger.info('N-Skart step 1')

    # initialize persistent environment for the algorithm
    env = dict()

    env['X_i'] = data
    env['N'] = data.size
    logger.debug('N = {}'.format(env['N']))
    if env['N'] < NSKART_MIN_DATA_POINTS:
        raise NSkartTooFewValues(
            'Need {} values, got {}.'.format(NSKART_MIN_DATA_POINTS, env['N'])
        )

    # From the given sample data set of size N, compute the sample skewness of
    # the last 80% of the observations.
    # Skewness is defined as in the scipy function here, as
    # G_1 = \frac{n^2}{(n-1)(n-2)} \frac{m_3}{s^3}
    sample_skewness = scipy.stats.skew(
        env['X_i'][math.floor(env['N'] / 5):], bias=False
    )
    logger.debug(
        'Sample skewness of the last 80% of the observations: {:.2f}'
        .format(sample_skewness)
    )

    # set initial batch size
    env['m'] = (
        1 if np.abs(sample_skewness) <= 4.0 else
        min(16, math.floor(env['N'] / 1280))
    )
    logger.debug(
        'Initial batch size: m = {}'.format(env['m'])
    )

    # set current number of batches in a spacer
    env['d'] = NSKART_INITIAL_NUMBER_OF_BATCHES_IN_SPACER

    # set maximum number of batches allowed in a spacer
    env['d^*'] = NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER

    # set nonspaced (adjacent) batch number
    env['k'] = NSKART_INITIAL_BATCH_NUMBER

    # set randomness test significance level
    env[NSKART_RANDOMNESS_TEST_SIGNIFICANCE_KEY] = (
        NSKART_RANDOMNESS_TEST_SIGNIFICANCE
    )

    # initialize randomness test counter
    env['b'] = 0

    # initially processed sample size
    env['n'] = env['k'] * env['m']

    # Having set an appropriate value for the initial batch size,
    # N-Skart uses the initial n = 1280m observations of the
    # overall sample of size N to compute k = 1280 nonspaced
    # (adjacent) batches of size m with an initial spacer consisting of
    # d ← 0 ignored batches preceding each “spaced” batch.
    env['Y_j(m)'] = _compute_nonspaced_batch_means(env)

    logger.debug('Post-step 1 environment: {}'.format(env))
    logger.info('Finish step 1')

    return env


def _step_2(env):
    """
    Perform step 2 of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)

    """

    yjs_sample_skewness = scipy.stats.skew(
        env['Y_j(m)'][math.ceil(0.8 * env['k']):], bias=False
    )
    logger.debug((
        'Sample skewness of the last 80% of the current set of nonspaced batch'
        ' means: {:.2f}'
    ).format(yjs_sample_skewness)
    )

    if abs(yjs_sample_skewness) > 0.5:
        # reset the maximum number of batches per spacer
        env['d^*'] = NSKART_MAXIMUM_NUMBER_OF_BATCHES_IN_SPACER_SKEWED
        logger.debug(
            'Reset the maximum number of batches per spacer to {}'.format(
                env['d^*']
            )
        )

    logger.debug('Post-step 2 environment: {}'.format(env))
    logger.info('Finish step 2')

    return env


def _step_3a(env):
    """
    Perform step 3a of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 3a')
    env[NSKART_NONSPACED_RANDOMNESS_TEST_KEY] = (
        von_neumann_ratio_test(
            data=env['Y_j(m)'],
            alpha=env[NSKART_RANDOMNESS_TEST_SIGNIFICANCE_KEY]
        )
    )
    if env[NSKART_NONSPACED_RANDOMNESS_TEST_KEY]:
        env["k'"] = env['k']

    logger.debug(
        'Randomness test for nonspaced batch means {}ed'.format(
            'pass' if env[NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
            else 'fail'
        )
    )
    logger.debug('Post-step 3a environment: {}'.format(env))
    logger.info('Finish step 3a')

    return env


def _step_3bd(env):
    """
    Perform step 3b/d of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 3b/d')

    # add another ignored batch to each spacer
    env['d'] += 1
    logger.debug('Add another ignored batch to each spacer: d = {}'.format(
        env['d']
    ))

    # number of spaced batches
    env["k'"] = math.floor(env['n'] / (env['d'] + 1) / env['m'])
    logger.debug("Number of spaced batches: k' = {}".format(env["k'"]))

    # compute spaced batch means
    env['Y_j(m,d)'] = env['Y_j(m)'][env['d']::env['d'] + 1]

    logger.debug('Post-step 3b/d environment: {}'.format(env))
    logger.info('Finish step 3b/d')
    return env


def _step_3c(env):
    """
    Perform step 3c of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 3c')
    env[NSKART_SPACED_RANDOMNESS_TEST_KEY] = (
        von_neumann_ratio_test(
            data=env['Y_j(m,d)'],
            alpha=env[NSKART_RANDOMNESS_TEST_SIGNIFICANCE_KEY]
        )
    )
    logger.debug(
        'Randomness test for spaced batch means {}ed'.format(
            'pass' if env[NSKART_SPACED_RANDOMNESS_TEST_KEY]
            else 'fail'
        )
    )

    logger.debug('Post-step 3c environment: {}'.format(env))
    logger.info('Finish step 3c')
    return env


def _step_4(env, continue_insufficient_data=False):
    """
    Perform step 4 of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 4')

    # tentative new batch size
    new_m = math.ceil(math.sqrt(2) * env['m'])

    # tentative new total batch count
    new_k = math.ceil(0.9 * env['k'])

    # tentative new processed sample size
    new_n = new_k * new_m

    if new_n <= env['N']:
        # update batch size
        env['m'] = new_m
        logger.debug('Update batch size: m = {}'.format(env['m']))

        # update total batch count
        env['k'] = new_k
        logger.debug('Update total batch count: k = {}'.format(env['k']))

        # update processed sample size
        env['n'] = new_n
        logger.debug('Update processed sample size: n = {}'.format(env['n']))

        # reset batch counter
        env['d'] = 0
        env['d^*'] = 10

        # increase test counter
        env['b'] += 1

        # recalculate non-spaced batch means
        env['Y_j(m)'] = _compute_nonspaced_batch_means(env)

    else:
        # insufficient data: sample to processed larger than available data
        logger.debug("Insufficient data.")
        logger.debug("New batch size: m = {}".format(new_m))
        logger.debug("New total batch count: k = {}".format(new_k))
        logger.debug("New processed sample size: n = {}".format(new_n))
        logger.debug("Available data points: N = {}".format(env['N']))

        error_msg = ("N = {} data points available, need n = {}".format(
            env['N'], new_n
        ))

        if continue_insufficient_data:
            logger.error(error_msg)
            logger.info("Insufficient data -- user request to continue")
            env[NSKART_INSUFFICIENT_DATA_KEY] = True

        else:
            raise NSkartInsufficientDataError(error_msg)

    logger.debug('Post-step 4 environment: {}'.format(env))
    logger.info('Finish step 4')
    return env


def _step_5a(env, **kwargs):
    """
    Perform step 5a of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 5a')

    # Skip first w observations in the warm-up periods
    env['w'] = env['d'] * env['m']

    # Number of approximately steady-state observations is available to build a
    # confidence interval
    env["N'"] = env['N'] - env['w']

    # Reinflate batch count to compensate for the total number of times the
    # batch count was deflated in successive iterations of the randomness test.
    env['k'] = min(
        math.ceil(env["k"] * (1.0 / 0.9) ** env['b']),
        env['k']
    )

    # compute a multiplier to use all available observations
    env['f'] = math.sqrt(env["N'"] / env["k'"] / env['m'])

    # update count of truncated, nonspaced batch means
    env["k'"] = min(math.floor(env['f'] * env["k'"]), 1024)

    # update batch size
    env['m'] = math.floor(
        env['f'] * env['m'] if env["k'"] < 1024
        else
        env["N'"] / 1024.0
    )

    # update length of warm-up period such that the initial w observations are
    # the only unused items in the overall data set of size N
    env['w'] += env["N'"] - env["k'"] * env['m']

    logger.debug('Post-step 5a environment: {}'.format(env))
    logger.info('Finish step 5a')
    return env


def _step_5b(env, **kwargs):
    """
    Perform step 5b of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 5b')

    # Recompute the truncated, nonspaced batch means so that there is no
    # partial batch left at the end of the overall data set of size N
    env['Y_j(m)'] = (
        env['X_i']
        [env['w']:]
        .reshape((env["k'"], env['m']))
        .mean(axis=1)
    )

    logger.debug('Post-step 5b environment: {}'.format(env))
    logger.info('Finish step 5b')
    return env


def _step_6(env, **kwargs):
    """
    Perform step 6 of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 6')

    # compute the grand average of the current set of truncated, nonspaced
    # batch means
    env[NSKART_GRAND_AVERAGE_KEY] = env['Y_j(m)'].mean()

    # compute the sample variance of the current set of truncated, nonspaced
    # batch means
    env[NSKART_SAMPLE_VAR_KEY] = env['Y_j(m)'].var(ddof=1)

    # compute the sample estimator of the lag-one correlation of the truncated,
    # nonspaced batch means
    env[NSKART_SAMPLE_LAG1_CORR] = (
        (
            (env['Y_j(m)'][:-1] - env[NSKART_GRAND_AVERAGE_KEY])
            * (env['Y_j(m)'][1:] - env[NSKART_GRAND_AVERAGE_KEY])
        )
        .sum()
        / env[NSKART_SAMPLE_VAR_KEY]
        / (env["k'"] - 1.0)
    )

    env['A'] = (
        (1. + env[NSKART_SAMPLE_LAG1_CORR])
        / (1. - env[NSKART_SAMPLE_LAG1_CORR])
    )

    logger.debug('Post-step 6 environment: {}'.format(env))
    logger.info('Finish step 6')
    return env


def _step_7(env, confidence_level=ONE_SIGMA, **kwargs):
    """
    Perform step 7 of the N-Skart algorithm

    Parameters
    ----------
    env: dict
        The persistent algorithm environment (parameters and variables)

    Returns
    -------
    env: dict
        The persistent algorithm environment (parameters and variables)
    """

    logger.info('N-Skart step 7')

    # compute spacer number such that spacer size is the smallest multiple of m
    # not less than the final size of the warm-up period
    env["d'"] = math.ceil(env['w'] / env['m'])

    # number of spaced batches
    env["k''"] = 1 + math.floor(
        (env["k'"] - 1) / (env["d'"] + 1)
    )

    # compute spaced batch means
    # FIXME: BUG in paper
    # Sum needs to start at: N - (k'' - j) (d' + 1) m - m + 1
    # Until: N - (k'' - j) (d' + 1) m
    mask = (
        np.arange(env['N'] - env['w']) // env['m'] % (env["d'"] + 1) == 0
    )[::-1]

    env["Y_j(m,d')"] = (
        env['X_i']
        [env['w']:]
        [mask]
        .reshape(
            (env["k''"], env['m'])
        )
        .mean(axis=1)
    )

    # compute grand mean
    env[NSKART_BATCHED_GRAND_MEAN_KEY] = env["Y_j(m,d')"].mean()

    # compute sample variance
    env[NSKART_BATCHED_SAMPLE_VAR_KEY] = env["Y_j(m,d')"].var(ddof=1)

    # compute skewness
    env[NSKART_BATCHED_SKEW_KEY] = scipy.stats.skew(
        env["Y_j(m,d')"], bias=False
    )

    env[r'\beta'] = env[NSKART_BATCHED_SKEW_KEY] / 6 / math.sqrt(env["k''"])

    def skewness_adjust(zeta):
        beta = env[r'\beta']
        return (
            (np.power(1 + 6 * beta * (zeta - beta), 1 / 3) - 1) / 2 / beta
        )

    env['G(zeta)'] = skewness_adjust
    env['L, R'] = np.asarray(
        scipy.stats.t(df=env["k''"] - 1)
        .interval(confidence_level)
    )
    env['CI'] = (
        env[NSKART_GRAND_AVERAGE_KEY]
        +
        env['G(zeta)'](env['L, R'])
        * math.sqrt(env['A'] * env[NSKART_SAMPLE_VAR_KEY] / env["k'"])
    )

    logger.debug('Post-step 7 environment: {}'.format(env))
    logger.info('Finish step 7')
    return env
