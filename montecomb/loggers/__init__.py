"""The loggers are responsible for computing metrics

A logger takes information from dataset evaluator calls, from behaviour
of agents, etc. and plots these metrics or returns those values. Examples
of metrics are number of calls, average value returned from calls, or
maximum of those, etc.
"""

import montecomb.loggers.counter
