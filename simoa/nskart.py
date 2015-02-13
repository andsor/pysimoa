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
import scipy.stats
from simoa.stats import von_neumann_ratio_test

logger = logging.getLogger(__name__)


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


class NSkartException(BaseException):
    pass


class NSkartInsufficientDataError(NSkartException, RuntimeError):
    pass


class NSkartTooFewValues(NSkartException, ValueError):
    pass


def compute_nskart_interval(
    data, confidence_level, continue_insufficient_data=True, verbose=False
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
    pass


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


def get_independent_data(xis, continue_insufficient_data=False, verbose=False):

    # STEP 1

    while True:
        # STEP 2

        # STEP 3

        # STEP 3a

        # randomness test failed (independence hypothesis rejected)

        while batch_number_in_spacer < max_batch_number_in_spacer:
            # STEP 3b/d

            # STEP 3c

            # Continue with Step 3b/d
            pass

        # STEP 4

        # continue with Step 2

    raise RuntimeError


one_sigma = scipy.stats.norm().cdf(1) - scipy.stats.norm().cdf(-1)


def nskart(
    data, confidence_level, continue_insufficient_data=True, verbose=False
):

    # STEP 5a
    # STEP 5b

    # STEP 6
    if verbose:
        print('Step 6')

    grand_average = yjs.mean()
    sample_var = yjs.var(ddof=1)
    sample_lag1_corr = (
        (yjs[:-1] - grand_average) * (yjs[1:] - grand_average)
        .sum()
        / sample_var
        / (batch_number - 1)
    )
    corr_adjustment = (1. + sample_lag1_corr) / (1. - sample_lag1_corr)

    if verbose:
        print(
            (
                "Grand average: {:.2f}\n"
                "Sample variance: {:.2f}\n"
                "Sample estimator of the lag-one correlation: {:.2f}\n"
                "Correlation adjustment: A = {:.2f}"
            ).format(
                grand_average,
                sample_var,
                sample_lag1_corr,
                corr_adjustment
            )
        )

    # STEP 7
    if verbose:
        print('Step 7: Confidence interval')

    ci_batch_number_in_spacer = math.ceil(initial_number / batch_size)
    # subtract 1 because of exceeding the sample size in some configurations
    # (bug in algorithm??)
    ci_batch_number = (
        1 + math.floor(
            (batch_number - 1) / (ci_batch_number_in_spacer + 1)
        ) - 1
    )

    ci_spaced_batch_means = (
        xis
        [:ci_batch_number * (ci_batch_number_in_spacer + 1) * batch_size]
        .reshape(
            (ci_batch_number * (ci_batch_number_in_spacer + 1), batch_size)
        )
        .mean(axis=1)
        [::-(ci_batch_number_in_spacer + 1)]
    )

    ci_spaced_batch_mean = ci_spaced_batch_means.mean()
    ci_spaced_batch_var = ci_spaced_batch_means.var(ddof=1)
    ci_spaced_batch_skew = scipy.stats.skew(ci_spaced_batch_means, bias=False)
    ci_beta = ci_spaced_batch_skew / 6. / math.sqrt(ci_batch_number)

    skewness_adjust = (
        lambda zeta: (
            (
                np.power(1. + 6. * ci_beta * (zeta - ci_beta), 1. / 3.) - 1.
            )
            / 2. / ci_beta
        )
    )
    critical_values = (
        np.asarray(
            scipy.stats.t(df=ci_batch_number - 1)
            .interval(confidence_level)
        )
    )
    ci = (
        grand_average
        +
        skewness_adjust(critical_values)
        * math.sqrt(corr_adjustment * sample_var / batch_number)
    )

    if verbose:
        print(
            (
                "Number of batches per spacer: d' = {}\n"
                "Batch number: k'' = {}\n"
                "Grand mean: {:.2f}\n"
                "Spaced batches sample variance: {:.2f}\n"
                "Spaced batches skewness: {:.2f}\n"
                "beta = {:.3f}"
            ).format(
                ci_batch_number_in_spacer,
                ci_batch_number,
                ci_spaced_batch_mean,
                ci_spaced_batch_var,
                ci_spaced_batch_skew,
                ci_beta,
            )
        )

    if verbose:
        print(
            (
                "Confidence level: {:.2f}%\n"
                "Unadjusted critical values: {:.2f}, {:.2f}\n"
                "Skewness-adjusted confidence interval: ({:.2f}, {:.2f})"
            ).format(
                confidence_level * 100,
                critical_values[0], critical_values[1],
                ci[0], ci[1]
            )
        )

    return independence, grand_average, ci
