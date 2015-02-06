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
import scipy.stats


def von_neumann_ratio_test(data, alpha, verbose=False):
    """
    Test a series of observations for independence (lack of autocorrelation)

    The von-Neumann ratio test is a statistical test designed for testing the
    independence of subsequent observations.
    The null hypothesis is that the data are independent and normally
    distributed.

    Parameters
    ----------
    alpha: float
        Significance level.
        This is the probability threshold below which the null hypothesis will
        be rejected.

    Notes
    -----
    Given a series :math:`x_i` of :math:`n` data points, the von-Neumann test
    statistic is [1]_ [6]_

    .. math::

       v = \frac{\sum_{i=1}^{n-1} (x_{i+1} - x_i)^2}{\sum_{i=1}^n (x_i -
           \bar{x})^2

    Under the null hypothesis, the mean :math:`\bar{v} = 2` and the variance
    :math:`\sigma^2_v = \frac{4 (n - 2)}{(n-1)(n+1)}` [3]_.



    References
    ----------

    .. [1] Von Neumann, J. (1941). Distribution of the ratio of the mean square
       successive difference to the variance. The Annals of Mathematical
       Statistics, 12(4), 367-395.

    .. [2] The Mean Square Successive Difference
       J. von Neumann, R. H. Kent, H. R. Bellinson and B. I. Hart
       The Annals of Mathematical Statistics
       Vol. 12, No. 2 (Jun., 1941) , pp. 153-162
       http://www.jstor.org/stable/2235765

    .. [3] Moments of the Ratio of the Mean Square Successive Difference to the
       Mean Square Difference in Samples From a Normal Universe J. D. Williams
       The Annals of Mathematical Statistics
       Vol. 12, No. 2 (Jun., 1941) , pp. 239-241
       http://www.jstor.org/stable/2235775

    .. [4] Madansky, Albert,
       Testing for Independence of Observations,
       In: Prescriptions for Working Statisticians
       Springer New York
       10.1007/978-1-4612-3794-5_4
       http://dx.doi.org/10.1007/978-1-4612-3794-5_4
    """

    mean_square_successive_difference = np.power(np.ediff1d(data), 2).mean()
    von_neumann_ratio = mean_square_successive_difference / data.var()
    von_neumann_mean = 2.
    von_neumann_var = 4. * (data.size - 2.) / (data.size ** 2 - 1.)
    acceptance_region = scipy.stats.norm.interval(
        1.0 - alpha,
        loc=von_neumann_mean,
        scale=np.sqrt(von_neumann_var)
    )

    if verbose:
        print(
            (
                "von Neumann ratio v = {:.2f},\n"
                "acceptance region at {:.0f}% significance level is "
                "({:.2f}, {:.2f})"
            ).format(
                von_neumann_ratio,
                alpha * 100,
                acceptance_region[0],
                acceptance_region[1]
            )
        )

    return acceptance_region[0] < von_neumann_ratio < acceptance_region[1]
