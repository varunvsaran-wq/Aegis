"""jevbench: an independent evaluation of Jev (TypeSafe AI's "System One" model).

Two studies share one interface (:class:`jevbench.deciders.Decider`):

- Study 1, prompt-injection screening (:mod:`jevbench.injection_data`).
- Study 4, confidence-gated model routing on RouterBench (:mod:`jevbench.routing`).

Terms of service: Jev outputs are used for evaluation only. Nothing in this
package trains a model on them (TypeSafe MCA section 2.3(b)).
"""

__version__ = "0.1.0"
