"""Module to make datasets to make decisions over

All classes in this datasets module allow you to choose an action,
or set thereof, and returns an evaluation of that action.
This evaluation is computed differently in each of these classes,
some are addition over subsets, some are local effects from a graph, etc.
"""

import montecomb.dataset.meta_dataset
import montecomb.dataset.subset_additions
import montecomb.dataset.local_effects
