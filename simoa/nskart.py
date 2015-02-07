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
import sys

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

    # Having set an appropriate value for the initial batch size,
    # N-Skart uses the initial n = 1280m observations of the
    # overall sample of size N to compute k = 1280 nonspaced
    # (adjacent) batches of size m with an initial spacer consisting of
    # d ← 0 ignored batches preceding each “spaced” batch.
    env['Y_j'] = (
        env['X_i']
        [:env['m'] * env['k']]
        .reshape((env['k'], env['m']))
        .mean(axis=1)
    )

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
        env['Y_j'][math.ceil(0.8 * env['k']):], bias=False
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
            data=env['Y_j'],
            alpha=env[NSKART_RANDOMNESS_TEST_SIGNIFICANCE_KEY]
        )
    )
    logger.debug(
        'Randomness test for nonspaced batch means {}ed'.format(
            'pass' if env[NSKART_NONSPACED_RANDOMNESS_TEST_KEY]
            else 'fail'
        )
    )
    logger.info('Finish step 3a')

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
            if verbose:
                print('Step 3b/d')

            batch_number_in_spacer += 1
            batch_number = math.floor(
                processed_sample_size
                / (batch_number_in_spacer + 1)
                / batch_size
            )
            spaced_batch_means = nonspaced_batch_means[
                batch_number_in_spacer::batch_number_in_spacer + 1
            ]
            assert(batch_number == spaced_batch_means.size)

            if verbose:
                print(
                    (
                        "Number of batches in spacer: d = {}\n"
                        "batch number: k' = {}"
                    ).format(
                        batch_number_in_spacer,
                        batch_number
                    )
                )

            # STEP 3c
            if verbose:
                print('Step 3c')

            if von_neumann_ratio_test(
                data=spaced_batch_means,
                alpha=randomness_test_significance_level,
                verbose=verbose
            ):
                # randomness test passed
                # (failed to reject independence hypothesis)
                if verbose:
                    print((
                        'Randomness test passed '
                        '(independence hypothesis failed-to-reject)'
                    ))
                return (
                    True,
                    processed_sample_size,
                    batch_size,
                    nonspaced_batch_number,
                    batch_number,
                    batch_number_in_spacer,
                    test_counter
                )

            # randomness test failed (independence hypothesis rejected)
            if verbose:
                print(
                    'Randomness test failed (independence hypothesis rejected)'
                )

            # Continue with Step 3b/d

        # STEP 4
        if verbose:
            print('Step 4')

        batch_size = math.ceil(math.sqrt(2) * batch_size)
        nonspaced_batch_number = math.ceil(0.9 * nonspaced_batch_number)
        processed_sample_size = nonspaced_batch_number * batch_size
        if processed_sample_size > sample_size:
            # insufficient data
            if verbose:
                print("Insufficient data.")

            if continue_insufficient_data:
                return (
                    False,
                    processed_sample_size,
                    batch_size,
                    nonspaced_batch_number,
                    batch_number,
                    batch_number_in_spacer,
                    test_counter
                )
            else:
                raise InsufficientDataError

        batch_number_in_spacer = 0
        max_batch_number_in_spacer = 10
        test_counter += 1

        if verbose:
            print(
                (
                    "Batch size: m = {}\n"
                    "current number of batches in spacer: d = {}\n"
                    "maximum number of batches allowed in a spacer: {}\n"
                    "batch number: k = {}\n"
                    "processed sample size: n = {}\n"
                    "number of times the batch count has been deflated "
                    "in the randomness test: b = {}"
                ).format(
                    batch_size,
                    batch_number_in_spacer,
                    max_batch_number_in_spacer,
                    nonspaced_batch_number,
                    processed_sample_size,
                    test_counter
                )
            )

        nonspaced_batch_means = (
            xis
            [:batch_size * batch_number]
            .reshape((batch_number, batch_size))
            .mean(axis=1)
        )
        assert(nonspaced_batch_means.size == batch_number)
        yjs = nonspaced_batch_means

        if verbose:
            sys.stdout.flush()

        # continue with Step 2

    raise RuntimeError


one_sigma = scipy.stats.norm().cdf(1) - scipy.stats.norm().cdf(-1)


def nskart(
    data, confidence_level, continue_insufficient_data=True, verbose=False
):

    xis = data

    (
        independence,
        processed_sample_size,
        batch_size,
        nonspaced_batch_number,
        batch_number,
        batch_number_in_spacer,
        test_counter
    ) = get_independent_data(
        xis,
        continue_insufficient_data=continue_insufficient_data,
        verbose=verbose
    )

    if verbose:
        if independence:
            print((
                "Failed to reject independence hypothesis: "
                "Passed randomness test."
            ))

        else:
            print("Insufficient data, failed randomness tests.")

        print(
            (
                "Processed sample size: n = {}\n"
                "batch size: m = {}\n"
                "(nonspaced) batch number: k = {}\n"
                "spaced batch number: k' = {}\n"
                "number of batches in spacer: d = {}\n"
                "number of times batch count has been deflated: b = {}"
            ).format(
                processed_sample_size,
                batch_size,
                nonspaced_batch_number,
                batch_number,
                batch_number_in_spacer,
                test_counter
            )
        )

    # STEP 5a
    if verbose:
        print('Step 5a')

    initial_number = batch_number_in_spacer * batch_size
    reduced_sample_size = xis.size - initial_number
    batch_number = min(
        math.ceil(batch_number * np.power(1. / 0.9, test_counter)),
        nonspaced_batch_number
    )
    inflation_factor = math.sqrt(
        reduced_sample_size / batch_number / batch_size
    )
    batch_number = min(math.floor(inflation_factor * batch_number), 1024)
    if batch_number < 1024:
        batch_size = math.floor(inflation_factor * batch_size)
    else:
        batch_size = math.floor(reduced_sample_size / 1024)

    initial_number += reduced_sample_size - batch_number * batch_size

    if verbose:
        print(
            (
                "Number of initial observations to skip: w = {}\n"
                "(inflated) batch number: k' = {}\n"
                "(inflated) batch size: m = {}\n"
                "inflation factor: f = {:.2f}"
            ).format(
                batch_number,
                batch_size,
                inflation_factor
            )
        )

    # STEP 5b
    if verbose:
        print('Step 5b')

    nonspaced_batch_means = (
        xis
        [initial_number:]
        .reshape((batch_number, batch_size))
        .mean(axis=1)
    )
    assert(nonspaced_batch_means.size == batch_number)
    yjs = nonspaced_batch_means

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
