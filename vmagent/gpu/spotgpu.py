"""SpotGPU as a set of orthogonal knobs, and the experiment variants as points in that space.

The engine used to have one hard-coded `spotgpu`, and the ten-scenario experiment suite had
five more hand-named policies in `experiments/preemption_20261107/harness.py`. Naming
worked as a research vocabulary but not as a coordinate system: `paper_cost` and
`legacy_spotgpu` differ in *two* axes at once (queue discipline and victim tie-break), so
their difference cannot be attributed, which is exactly what that suite's own report says
about it. This module replaces the names with the axes they were secretly standing for.

Six knobs, defaults reproducing the engine's historical `spotgpu` exactly:

    preemption              bool   whether HP tasks may evict LP spot tasks
    victim_selection        cost | random      OSDI'26 §4.2 uses cost; random is the
                                  paper's own baseline control
    victim_prefix           joint | gpu        the paper scores a node by the smallest
                                  prefix with sum(G_t) >= G_hp, i.e. GPU only. The
                                  engine has always additionally required CPU, so
                                  `joint` is the historical behaviour and `gpu` is
                                  the literal paper reading. Default keeps history;
                                  the difference is measured, not asserted.
    hp_first_queue          bool   search HP GPU candidates before ordinary backfill.
                                  The paper does not specify a global queue rule, so
                                  this is a local experimental choice.
    placement_terms         tuple over (packing, co_locate, history), applied in that
                                  order, each breaking the previous tie. All three is
                                  the paper's §4.2.1; dropping the tail gives the
                                  paper's ablations.
    placement_cpu_aware     none | sum | lexicographic
                                  The paper ranks by idle GPU capacity only. `sum`
                                  matches this engine's own best_fit (normalised
                                  GPU+CPU leftover), `lexicographic` matches the
                                  defragmentation benchmark's best_fit (GPU leftover
                                  first, CPU only as a tie-break). Three definitions
                                  of "least leftover" now live in one policy, so a
                                  reader can see which one a number came from.
"""

VICTIM_SELECTIONS = ("cost", "random")
VICTIM_PREFIXES = ("joint", "gpu")
CPU_AWARE_MODES = ("none", "sum", "lexicographic")
TERMS = ("packing", "co_locate", "history")

DEFAULTS = {"preemption": True, "victim_selection": "cost", "victim_prefix": "joint",
            "hp_first_queue": False, "placement_terms": TERMS, "placement_cpu_aware": "none"}

# The five experiment policies, expressed as knob values rather than as code branches.
# `legacy_spotgpu` is the empty override: it is what the engine has always done.
VARIANTS = {
    "legacy_spotgpu": {},
    "paper_cost": {"hp_first_queue": True},
    "no_preempt": {"hp_first_queue": True, "preemption": False},
    "random_victim": {"hp_first_queue": True, "victim_selection": "random"},
    "paper_baseline": {"hp_first_queue": True, "victim_selection": "random",
                       "placement_terms": ("packing",)},
    "no_history": {"hp_first_queue": True, "placement_terms": ("packing", "co_locate")},
}


def resolve(values=None):
    """Validate a knob mapping and return the complete set."""
    values = dict(values or {})
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError("Unknown SpotGPU option(s): " + ", ".join(unknown))
    knobs = dict(DEFAULTS)
    knobs.update(values)
    if knobs["victim_selection"] not in VICTIM_SELECTIONS:
        raise ValueError("victim_selection must be cost or random")
    if knobs["victim_prefix"] not in VICTIM_PREFIXES:
        raise ValueError("victim_prefix must be joint or gpu")
    if knobs["placement_cpu_aware"] not in CPU_AWARE_MODES:
        raise ValueError("placement_cpu_aware must be none, sum or lexicographic")
    terms = tuple(knobs["placement_terms"])
    if terms != TERMS[:len(terms)]:
        raise ValueError("placement_terms must be a prefix of " + str(TERMS))
    knobs["placement_terms"] = terms
    for key in ("preemption", "hp_first_queue"):
        knobs[key] = bool(knobs[key])
    return knobs


def variant(name):
    if name not in VARIANTS:
        raise ValueError("Unknown SpotGPU variant: " + str(name))
    return resolve(VARIANTS[name])
