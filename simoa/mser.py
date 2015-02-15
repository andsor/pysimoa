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

logger = logging.getLogger(__name__)


# check python 3
if not (3 / 2 == 1.5):
    raise RuntimeError


ONE_SIGMA = scipy.stats.norm().cdf(1) - scipy.stats.norm().cdf(-1)


MSER5_TRUNCATED_MEAN_KEY = r'\bar{Z}(k,d^*)'
MSER5_TRUNCATED_VARIANCE_KEY = r'S^2_Z(k,d)'
MSER5_NEW_BATCH_MEANS_KEY = r'\bar{Z}_l(m^*,d^*)'
MSER5_NEW_BATCH_MEANS_VARIANCE_KEY = r'S^2_{\bar{Z}}(k^*, m^*, d^*)'


class MSERException(BaseException):
    pass


class MSERInsufficientDataError(MSERException, RuntimeError):
    pass


MSERReturnValue = namedtuple(
    'MSERReturnValue',
    ['mean', 'ci', 'env']
)


def compute_mser5_interval(
    data, confidence_level=ONE_SIGMA, continue_insufficient_data=False
):
    """
    Estimate the steady-state mean and the interval by the MSER-5 method

    Notes
    -----
    "MSER-5 (Franklin and White 2008) is a modification of the Marginal
    Confidence Rule (MCR) or the Marginal Standard Error Rule (MSER) proposed
    by White (1997)." [1]_
    "MSER-5 is a sequential procedure that uses the method of nonoverlapping
    batch means (NBM) to deliver not only a data-truncation point (i.e., end of
    the “warm-up period”) but also point and CI estimators [...] computed from
    the truncated (“warmed-up”) sequence of batch means." [1]_

    References
    ----------
    .. [1] Mokashi, A. C. et al. Performance comparison of MSER-5 and N-Skart
       on the simulation start-up problem. In *Proceedings of the Winter
       Simulation Conference*, 971-982 (2010).
       http://www.ise.ncsu.edu/jwilson/files/mokashi10wsc.pdf
    .. [2] Franklin, W. W. & White, K. P. Stationarity tests and MSER-5:
       Exploring the intuition behind mean-squared-error-reduction in
       detecting and correcting initialization bias. In Winter Simulation
       Conference, 541-546 (2008).
       http://www.informs-sim.org/wsc08papers/064.pdf
    .. [3] White, K. P. An effective truncation heuristic for bias reduction in
       simulation output. Simulation 69, 323-334 (1997).
       http://dx.doi.org/10.1177/003754979706900601
    """

    env = dict()

    env[r'1 - \alpha'] = confidence_level

    env['X_i'] = data

    # number of data points
    env['N'] = env['X_i'].size

    # batch size
    env['m'] = 5

    # number of batches
    env['k'] = int(math.floor(env['N'] / env['m']))

    # compute batch means
    env['Z_j'] = (
        env['X_i']
        [:env['m'] * env['k']]
        .reshape(env['k'], env['m'])
        .mean(axis=1)
    )

    env['objective'] = np.empty(env['k'] - 5)
    for d in range(env['k'] - 5):
        # compute objective function
        env['objective'][d] = (
            env['Z_j'][d:env['k']].var(ddof=0)
            / (env['k'] - d)
        )

    # compute truncation point
    env['d^*'] = env['objective'].argmin()

    if env['d^*'] >= math.floor(env['k'] / 2):
        # insufficient data
        error_msg = (
            'Insufficient data: truncation point at {}, should be below {}'
        ).format(env['d^*'], math.floor(env['k'] / 2))

        if continue_insufficient_data:
            logger.error(error_msg)
        else:
            raise MSERInsufficientDataError(error_msg)

    # compute truncated mean
    env[MSER5_TRUNCATED_MEAN_KEY] = env['Z_j'][env['d^*']:].mean()

    # compute truncated sample variance
    env[MSER5_TRUNCATED_VARIANCE_KEY] = env['Z_j'][env['d^*']:].var(ddof=0)

    """
    env['k^*'] = 20
    env['m^*'] = int(math.floor((env['k'] - env['d^*']) / env['k^*']))

    # To compute a CI estimator, MSER-5 organizes the truncated sequence
    # into 20 “new” batch means with batch size as suggested by White and
    # Robinson (2009)
    env[MSER5_NEW_BATCH_MEANS_KEY] = (
        env['Z_j'][
            env['d^*']:
            env['d^*'] + env['k^*'] * env['m^*']
        ]
        .reshape(env['k^*'], env['m^*'])
        .mean(axis=1)
    )

    # compute sample variance of the new batch means
    env[MSER5_NEW_BATCH_MEANS_VARIANCE_KEY] = (
        env[MSER5_NEW_BATCH_MEANS_KEY].var(ddof=1)
    )
    """

    critical_values = np.asarray(
        scipy.stats.norm.interval(confidence_level)
    )

    env['CI'] = (
        env[MSER5_TRUNCATED_MEAN_KEY]
        + critical_values
        * env[MSER5_TRUNCATED_VARIANCE_KEY]
        / math.sqrt(env['k'] - env['d^*'])
    )

    return MSERReturnValue(
        mean=env[MSER5_TRUNCATED_MEAN_KEY],
        ci=env['CI'],
        env=env,
    )
