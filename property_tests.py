"""
E0 251o — Property-based tests for NetworkX (Hypothesis).

Team: Abhishek Malik
SNo: 13-19-01-19-52-25-1-26268
Team Size: 1

Algorithms tested:
  - Node classification: harmonic_function (Gaussian fields / harmonic relaxation),
    local_and_global_consistency (networkx.algorithms.node_classification)
  - Graph hashing: nx.weisfeiler_lehman_graph_hash (Weisfeiler–Lehman, WL)
"""

from __future__ import annotations

import warnings

import networkx as nx
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from networkx.algorithms import node_classification


# Helper Functions


def wl_hash(graph, **kwargs):
    """Call WL hash suppressing known NetworkX UserWarnings (degree-based / directed hashing)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(graph, **kwargs)


@st.composite
def undirected_simple_graphs(draw):
    """Random simple undirected G(n,p) graphs; may be disconnected."""
    n = draw(st.integers(min_value=2, max_value=14))
    p = draw(st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return nx.gnp_random_graph(n, p, seed=seed)


@st.composite
def directed_simple_graphs(draw):
    """Random simple directed graphs (no multi-arcs); may be weakly disconnected."""
    n = draw(st.integers(min_value=2, max_value=12))
    p = draw(st.floats(min_value=0.08, max_value=0.92, allow_nan=False, allow_infinity=False))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return nx.gnp_random_graph(n, p, seed=seed, directed=True)


@st.composite
def undirected_with_edge_labels(draw):
    """
    Connected simple undirected graphs where every edge carries a short string label.
    Used to test WL with edge_attr (structured / weighted-style labels).
    """
    n = draw(st.integers(min_value=3, max_value=12))
    p = draw(st.floats(min_value=0.2, max_value=0.95, allow_nan=False, allow_infinity=False))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    g = nx.gnp_random_graph(n, p, seed=seed)
    assume(nx.is_connected(g))
    labels = draw(
        st.lists(
            st.sampled_from(["a", "b", "c", "x"]),
            min_size=g.number_of_edges(),
            max_size=g.number_of_edges(),
        )
    )
    for (u, v), lab in zip(g.edges(), labels):
        g[u][v]["label"] = lab
    return g


def _random_relabeling_map(draw, nodes: list):
    """Random permutation relabeling (bijection on the node set)."""
    n = len(nodes)
    perm = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return {nodes[i]: nodes[perm[i]] for i in range(n)}


NC_LABEL = "label"


@st.composite
def connected_graph_semi_supervised(draw):
    """
    Connected undirected simple graphs with at least one seeded node and at least one
    unlabeled node (semi-supervised). Labels use attribute NC_LABEL.
    Mixes path/cycle topologies with G(n,p) random graphs for diversity.
    """
    kind = draw(st.sampled_from(["er", "path", "cycle"]))
    if kind == "path":
        n = draw(st.integers(min_value=4, max_value=18))
        g = nx.path_graph(n)
    elif kind == "cycle":
        n = draw(st.integers(min_value=4, max_value=16))
        g = nx.cycle_graph(n)
    else:
        n = draw(st.integers(min_value=4, max_value=16))
        p = draw(st.floats(min_value=0.15, max_value=0.95, allow_nan=False, allow_infinity=False))
        seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
        g = nx.gnp_random_graph(n, p, seed=seed)
        assume(nx.is_connected(g))
    nodes = list(g.nodes())
    n_labeled = draw(st.integers(min_value=1, max_value=n - 1))
    labeled = draw(
        st.lists(st.sampled_from(nodes), min_size=n_labeled, max_size=n_labeled, unique=True)
    )
    for v in labeled:
        g.nodes[v][NC_LABEL] = draw(st.sampled_from(["A", "B", "C"]))
    return g


@st.composite
def connected_graph_single_class_seeds(draw):
    """Connected graph; every seed uses the same class label (boundary for label vocabulary)."""
    kind = draw(st.sampled_from(["er", "path"]))
    if kind == "path":
        n = draw(st.integers(min_value=3, max_value=16))
        g = nx.path_graph(n)
    else:
        n = draw(st.integers(min_value=3, max_value=14))
        p = draw(st.floats(min_value=0.2, max_value=0.95, allow_nan=False, allow_infinity=False))
        seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
        g = nx.gnp_random_graph(n, p, seed=seed)
        assume(nx.is_connected(g))
    nodes = list(g.nodes())
    n_labeled = draw(st.integers(min_value=1, max_value=n - 1))
    labeled = draw(
        st.lists(st.sampled_from(nodes), min_size=n_labeled, max_size=n_labeled, unique=True)
    )
    for v in labeled:
        g.nodes[v][NC_LABEL] = "only"
    return g


@st.composite
def connected_fully_labeled_graph(draw):
    """Every vertex has NC_LABEL; graph is connected (path, cycle, or G(n,p))."""
    kind = draw(st.sampled_from(["er", "path", "cycle"]))
    if kind == "path":
        n = draw(st.integers(min_value=4, max_value=16))
        g = nx.path_graph(n)
    elif kind == "cycle":
        n = draw(st.integers(min_value=4, max_value=14))
        g = nx.cycle_graph(n)
    else:
        n = draw(st.integers(min_value=4, max_value=14))
        p = draw(st.floats(min_value=0.25, max_value=0.95, allow_nan=False, allow_infinity=False))
        seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
        g = nx.gnp_random_graph(n, p, seed=seed)
        assume(nx.is_connected(g))
    for v in g.nodes():
        g.nodes[v][NC_LABEL] = draw(st.sampled_from(["pos", "neg"]))
    return g


# Node classification: six property-based tests


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(connected_graph_semi_supervised())
def test_harmonic_output_length_and_label_vocabulary(g):
    """
    Property: `harmonic_function` returns one predicted class per node, in the same order
    as `list(G.nodes())`, and every predicted class is one of the classes that already
    appear on seeded nodes.

    Why it matters: Semi-supervised classifiers should not invent new label strings.
    Callers align predictions with `G.nodes()` iteration order; wrong length or mystery
    labels would break training pipelines, evaluation joins, and any code that zips nodes
    with predictions.

    Mathematical basis: The implementation builds a fixed set of class columns from the
    seeds only (`label_dict`), then predicts by `argmax` over those columns. So outputs
    live in the same finite alphabet as the seeds.

    Test strategy: Connected graphs from paths, cycles, or G(n,p) models; random non-empty
    proper subset of labeled vertices with labels in {A, B, C}, leaving some unlabeled.

    Assumptions / preconditions: Undirected simple graph (required by NetworkX);
    at least one node carries `NC_LABEL`; `numpy`/`scipy` available. We use `max_iter=80`.

    Why this matters (failure): Wrong length means node–prediction alignment is broken.
    A prediction outside the seed vocabulary would mean `argmax` indexing or `label_dict`
    handling is inconsistent with the documented contract.
    """
    nodes = list(g.nodes())
    seed_values = {g.nodes[v][NC_LABEL] for v in nodes if NC_LABEL in g.nodes[v]}
    pred = node_classification.harmonic_function(g, label_name=NC_LABEL, max_iter=80)
    assert len(pred) == len(nodes)
    for y in pred:
        assert y in seed_values


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(connected_graph_semi_supervised())
def test_harmonic_preserves_seed_labels_on_labeled_nodes(g):
    """
    Property (invariant): Any vertex that already has a seed label must receive exactly
    that same label in the harmonic-function prediction list.

    Why it matters: Labeled nodes are *constraints* in Gaussian-field / harmonic
    formulations—they should act as pinned boundary values. If a seed were overwritten,
    the routine would violate the semi-supervised meaning of “known labels.”

    Mathematical basis: In the reference iteration, rows of the propagation operator
    corresponding to labeled nodes are zeroed and a one-hot clamp term is added so those
    coordinates stay fixed at the seed class under the update `F <- P @ F + B`.

    Test strategy: Same semi-supervised random connected graphs as other NC tests; compare
    each seeded vertex’s attribute to the entry at the same index as in `list(G.nodes())`.

    Assumptions / preconditions: Undirected graph; default harmonic iteration count
    sufficient for numerical stability on these small graphs (we use max_iter=120).

    Why this matters (failure): A changed seed would indicate broken clamping or wrong
    row indexing—silent corruption of ground-truth labels used for evaluation or active
    learning.
    """
    nodes = list(g.nodes())
    pred = node_classification.harmonic_function(g, label_name=NC_LABEL, max_iter=120)
    for i, v in enumerate(nodes):
        if NC_LABEL in g.nodes[v]:
            assert pred[i] == g.nodes[v][NC_LABEL]


@settings(deadline=None, max_examples=35, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(connected_fully_labeled_graph())
def test_harmonic_on_fully_labeled_graph_recover_ground_truth(g):
    """
    Property (postcondition / boundary): If every vertex is already labeled, harmonic
    propagation should return exactly those labels (graph already fully observed).

    Why it matters: When there is nothing left to infer, the algorithm should reduce to
    the identity on labels; this is a sanity check that the iterative scheme respects
    fully supervised corner cases.

    Mathematical basis: With all nodes labeled and fixed rows in the update, the unique
    sensible solution matches the given labels on every node; the implementation’s clamp
    forces labeled coordinates to their one-hot seeds.

    Test strategy: Paths, cycles, or connected G(n,p) graphs; each vertex gets `pos` or `neg`;
    run harmonic with a generous `max_iter` so numerical iteration has settled on small graphs.

    Assumptions / preconditions: Connected undirected graph; `NC_LABEL` on every node.

    Why this matters (failure): Any mismatch would show incorrect handling of the
    all-labeled regime (indexing, label encoding, or iteration) even before considering
    harder partially labeled cases.
    """
    nodes = list(g.nodes())
    pred = node_classification.harmonic_function(g, label_name=NC_LABEL, max_iter=400)
    for i, v in enumerate(nodes):
        assert pred[i] == g.nodes[v][NC_LABEL]


@settings(deadline=None, max_examples=45, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(connected_graph_single_class_seeds())
def test_lgc_single_seed_class_propagates_to_all_nodes(g):
    """
    Property (boundary / invariant): If every seeded node carries the same class string,
    `local_and_global_consistency` assigns that same class to every vertex.

    Why it matters: With only one class in the seed vocabulary, there is no alternative
    column for `argmax` to select. All nodes should collapse to that class—otherwise the
    method would be inventing structure not present in the label space.

    Mathematical basis: `label_dict` contains a single label ID; the score matrix has one
    column; after propagation the argmax column is always 0, so every node maps back to
    the sole label.

    Test strategy: Paths or connected G(n,p) graphs; a non-empty proper subset of nodes labeled
    `"only"` and the rest unlabeled.

    Assumptions / preconditions: Undirected graph; `local_and_global_consistency` default
    `alpha` (0.99) and `max_iter` large enough for these instances.

    Why this matters (failure): Any other predicted string would mean the class dictionary
    or argmax decoding disagrees with a trivial one-class setup.
    """
    pred = node_classification.local_and_global_consistency(g, label_name=NC_LABEL, max_iter=120)
    assert all(y == "only" for y in pred)


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(connected_graph_semi_supervised())
def test_lgc_output_length_and_label_vocabulary(g):
    """
    Property: `local_and_global_consistency` returns one label per node in `list(G.nodes())`
    order, and every predicted label comes from the seed vocabulary.

    Why it matters: The public API mirrors harmonic's list output; callers need correct
    length and no invented classes. Unlike `harmonic_function`, this algorithm *does not*
    zero propagation on labeled rows—seeds are soft-injected via `B` each step—so
    **seed rows are not guaranteed to stay identically equal to their original strings**
    at `argmax`. Testing that stronger property would be wrong; vocabulary + length *are*
    part of the documented scoring setup (argmax over seeded classes only).

    Mathematical basis: Class columns are built only from seeds (`label_dict`); `F` has
    one column per seed class, so `argmax` never selects an out-of-vocabulary label.

    Test strategy: Same semi-supervised path/cycle/G(n,p) graphs as the harmonic vocabulary test.

    Assumptions / preconditions: Undirected graph; `max_iter=120`; default `alpha=0.99`.

    Why this matters (failure): Wrong length breaks alignment with `G.nodes()`. A prediction
    outside the seed set would break the column construction contract.
    """
    nodes = list(g.nodes())
    seed_values = {g.nodes[v][NC_LABEL] for v in nodes if NC_LABEL in g.nodes[v]}
    pred = node_classification.local_and_global_consistency(g, label_name=NC_LABEL, max_iter=120)
    assert len(pred) == len(nodes)
    for y in pred:
        assert y in seed_values


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(connected_graph_semi_supervised())
def test_harmonic_longer_iteration_does_not_change_argmax_labels(g):
    """
    Property (metamorphic / idempotence-style): After enough iterations, increasing
    `max_iter` further leaves the *discrete* predicted labels unchanged on these small
    graphs (we compare a fairly deep budget against a much deeper one).

    Why it matters: Users should not get different class assignments from merely raising
    the iteration cap after convergence—otherwise results would be arbitrary functions of
    `max_iter` instead of the underlying linear fixed-point problem.

    Mathematical basis: The update is a stationary linear iteration toward a fixed point;
    after convergence (in exact arithmetic) further steps do not change `F`, hence `argmax`
    is stable. We test the practical analogue on small graphs with two iteration budgets.

    Test strategy: Same semi-supervised random connected graphs as elsewhere; compare string
    predictions for `max_iter=250` vs `max_iter=1000` so the shorter run is likelier to
    have reached a practical fixed point on these small graphs.

    Assumptions / preconditions: Undirected connected graphs with the usual seeding; float
    arithmetic may rarely tie at `argmax`—if a counterexample appeared, it would point to
    marginal numerical instability at class boundaries.

    Why this matters (failure): Different label lists for deeper iteration would suggest
    non-stationary behavior or bugs in the loop (e.g. accidental state reuse across calls)
    rather than genuine model ambiguity.
    """
    a = node_classification.harmonic_function(g, label_name=NC_LABEL, max_iter=250)
    b = node_classification.harmonic_function(g, label_name=NC_LABEL, max_iter=1000)
    assert a == b


# Weisfeiler–Lehman graph hash: five property-based tests


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    st.one_of(
        undirected_simple_graphs(),
        directed_simple_graphs(),
        st.just(nx.Graph()),
        st.builds(lambda n: nx.path_graph(n), st.integers(min_value=1, max_value=8)),
    ),
    st.integers(min_value=8, max_value=32),
)
def test_wl_hash_is_hex_string_of_expected_length(g, digest_size):
    """
    Property: `weisfeiler_lehman_graph_hash` returns a hexadecimal string whose length is
    exactly twice the chosen `digest_size` (bytes → hex digits).

    Why it matters: Callers often store, compare, or log hashes as strings; a stable,
    documented format (hex of fixed width) is part of the public contract. Random
    bytes or wrong length would break databases, equality checks, or UI display.

    Mathematical basis: The implementation uses BLAKE2b digests and exposes them as
    hex; each byte encodes as two hex characters.

    Test strategy: Mix inputs—dense/sparse G(n,p) undirected graphs, directed
    random graphs, the empty graph, and path graphs (structured tree-like topology)—
    and sweep `digest_size` over a safe range.

    Assumptions / preconditions: `iterations` is positive (fixed internally in each
    call to 3 here); graph is not a MultiGraph (NetworkX API). Warnings from
    unlabeled hashing are ignored in the helper— they do not affect the length check.

    Why this matters (failure): Wrong length or non-hex output would indicate broken
    hashing, wrong digest_size handling, or corrupted return type.
    """
    h = wl_hash(g, iterations=3, digest_size=digest_size)
    assert len(h) == 2 * digest_size
    int(h, 16)  # raises if not valid hex


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(undirected_simple_graphs())
def test_wl_hash_is_idempotent_on_repeated_calls(g):
    """
    Property (idempotence): Calling the WL graph hash twice on the same graph with the
    same parameters returns the same string.

    Why it matters: A hash function used for indexing, regression tests, or caching
    must be deterministic. Non-determinism would make flaky tests and unstable
    fingerprints.

    Mathematical basis: The WL procedure in NetworkX is deterministic for fixed G and
    fixed hyperparameters (iterations, digest_size, attribute keys).

    Test strategy: Random undirected graphs of varying size and density; compare two
    consecutive calls with identical arguments.

    Assumptions / preconditions: Same as the WL function (simple graph; not a
    multigraph). Warnings suppressed in helper.

    Why this matters (failure): Different strings on repeat would imply hidden random
    state, race conditions, or non-deterministic iteration order where the algorithm
    should be fixed—serious for any reproducible workflow.
    """
    a = wl_hash(g, iterations=3, digest_size=16)
    b = wl_hash(g, iterations=3, digest_size=16)
    assert a == b


@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
@given(undirected_simple_graphs(), st.data())
def test_wl_hash_invariant_under_vertex_relabeling(g, data):
    """
    Property (metamorphic): If H is obtained from G by renaming vertices with a
    bijection (same structure, different labels), the WL graph hash is unchanged.

    Why it matters: A useful graph hash should not depend on arbitrary vertex IDs.
    **Relabeling-isomorphism** (same graph, renamed vertices) must yield the same digest.
    Note: different non-isomorphic graphs can still share a WL hash (collisions exist);
    this test does *not* claim the hash is a perfect isomorphism certificate.

    Mathematical basis: One step of WL refines labels from multisets of neighbors; an
    isomorphism preserves those multisets at every round, so isomorphic graphs stay in
    lockstep. The final multiset of refined labels—and thus this hash—is unchanged
    after a consistent vertex rename.

    Test strategy: Random simple undirected graphs; apply a random permutation
    relabeling with `relabel_nodes`; compare WL hashes with fixed iterations and
    digest_size.

    Assumptions / preconditions: G is simple, undirected, unlabeled (default
    degree-based initialization). No edge_attr/node_attr so both graphs use the same
    labeling scheme.

    Why this matters (failure): Different hashes after a pure relabeling would mean the
    fingerprint encodes node names, not just structure—defeating the purpose of graph
    hashing for anonymized or permuted data.
    """
    nodes = list(g.nodes())
    mapping = _random_relabeling_map(data.draw, nodes)
    h = nx.relabel_nodes(g, mapping, copy=True)
    assert wl_hash(g, iterations=3, digest_size=16) == wl_hash(h, iterations=3, digest_size=16)


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.filter_too_much])
@given(undirected_with_edge_labels(), st.data())
def test_wl_hash_invariant_under_relabeling_when_edges_are_labeled(g, data):
    """
    Property (metamorphic): With a fixed edge attribute key (`edge_attr='label'`),
    isomorphic relabeling preserving those labels yields the same WL hash.

    Why it matters: Many applications encode types of relations or strengths on edges.
    The hash must still be invariant under consistent renaming of endpoints so that
    “same interaction pattern, different user IDs” maps to one fingerprint. As with the
    unlabeled case, **non-isomorphic** graphs may occasionally collide; we only assert
    invariance under **isomorphic relabeling** with preserved edge attributes.

    Mathematical basis: Edge-tagged WL prefixes neighbor contributions with the attribute
    on the connecting edge; an isomorphism carries both endpoints and those tags, so
    each node’s aggregated label string matches after rename.

    Test strategy: Random connected graphs with small string labels on every edge;
    apply a random vertex permutation via `relabel_nodes` (NetworkX copies edge data).

    Assumptions / preconditions: Connected graph so hashing uses a non-trivial pattern;
    every edge has attribute `"label"`.

    Why this matters (failure): Mismatch after relabeling with identical edge tags would
    mean the fingerprint depends on vertex IDs rather than the attributed structure alone.
    """
    nodes = list(g.nodes())
    mapping = _random_relabeling_map(data.draw, nodes)
    h = nx.relabel_nodes(g, mapping, copy=True)
    assert wl_hash(g, iterations=3, digest_size=16, edge_attr="label") == wl_hash(
        h, iterations=3, digest_size=16, edge_attr="label"
    )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: nx.Graph(), id="empty_graph"),
        pytest.param(lambda: nx.path_graph(1), id="single_vertex_path"),
    ],
)
def test_wl_hash_extreme_small_graphs_are_stable(factory):
    """
    Property (boundary / postcondition): On the empty graph and on a single isolated
    vertex, the WL hash is still well-defined, deterministic, and matches on
    repeated calls.

    Why it matters: Real pipelines may start from degenerate inputs (empty graph after
    filtering, one remaining entity). The API should not crash and should give stable
    outputs suitable for testing or logging.

    Mathematical basis: With no vertices, the multiset of refined labels is empty; with
    one vertex, there are no neighbors, so WL rounds still deterministically map to
    hashes of multiset counters. Idempotence remains a basic expectation.

    Test strategy: Explicit constructions for the empty graph and for a one-node path;
    compare two consecutive hashes for each.

    Assumptions / preconditions: Positive `iterations` after NetworkX internal
    adjustment for unlabeled graphs; default WL settings otherwise.

    Why this matters (failure): Crashes, length violations, or changing outputs on
    repeat would break smoke tests and violate determinism on minimal inputs.
    """
    g = factory()
    h1 = wl_hash(g, iterations=2, digest_size=16)
    h2 = wl_hash(g, iterations=2, digest_size=16)
    assert h1 == h2
    assert len(h1) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
