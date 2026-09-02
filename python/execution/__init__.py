"""Strategy-agnostic execution layer.

Consumes target weights and knows nothing about regimes: a shadow decision row
becomes whole-share orders against a broker behind one interface. The paper
broker is a local cash-account simulation whose journal is git-tracked and
append-only, the same contract as the shadow log. Promotion of any strategy to
this layer is a config change, never an execution change.
"""
