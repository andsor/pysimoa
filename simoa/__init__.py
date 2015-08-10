# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import (ascii, bytes, chr, dict, filter, hex, input,
                             int, map, next, oct, open, pow, range, round,
                             str, super, zip)

from .nskart import (
    compute_nskart_interval,
    NSkartException,
    NSkartTooFewValues,
    NSkartInsufficientDataError,
)

from .mser import (
    compute_mser5_interval,
    MSERException,
    MSERInsufficientDataError,
)


import pkg_resources

try:
        __version__ = pkg_resources.get_distribution(__name__).version
except:
        __version__ = 'unknown'
