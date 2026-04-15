# E0 251o — NetworkX property-based tests

This project uses **Hypothesis** to write **property-based tests** for **NetworkX**.  
Hypothesis builds many random graphs (different sizes, how dense they are, and how labels are placed). Each test states a rule that should stay true for all of those examples.

## Algorithms we test

### 1. Node classification (semi-supervised on graphs)

NetworkX exposes this in `networkx.algorithms.node_classification` (not on the short `nx.` path). We use:

- **`harmonic_function`** — classic **harmonic / Gaussian-field** semi-supervised labeling on the graph.
- **`local_and_global_consistency`** — **local and global consistency** label propagation (balances local smoothing with a global constraint).

In plain words: a few vertices have known **class labels**; the rest are **unlabeled**. The algorithm spreads information along edges to guess labels for everyone. The tests check things like:

- You get **one prediction per vertex**, in the same order as **`list(G.nodes())`**.
- Predictions only use **label strings that already appear** on seeded vertices (no mystery classes).
- **`harmonic_function`** keeps **seed vertices** on their **original** labels (hard boundary). **`local_and_global_consistency`** still only predicts from the **seed vocabulary**, but it **soft-injects** seeds, so a seed’s row is **not** guaranteed to stay equal to its input string—that is **not** a bug; we do not test that stronger claim for LGC.
- If **every** vertex is already labeled, **`harmonic_function`** should return those labels (we test that). LGC uses a different update and is not asserted to match that same identity here.
- If seeds only use **one** class, **every** vertex should end up with that class under `local_and_global_consistency`.
- Running the harmonic method with **more iterations** should not flip discrete labels once things have settled (stability of the answer, not of floating-point noise).

Graphs here are **undirected** and **connected**. We mix **paths**, **cycles**, and **G(n,p) random graphs** (also called binomial random graphs) so tests see different shapes, not only random density. Setups are **semi-supervised** (some labeled, some not) or **fully labeled** where needed. The routines need **NumPy** and **SciPy** under the hood.

### 2. Graph hashing (Weisfeiler–Lehman)

We use **`nx.weisfeiler_lehman_graph_hash`**. It produces a **fingerprint** string for a graph. The tests check format, determinism, invariance when you **relabel** vertices, behavior with **edge labels**, and **tiny** graphs (empty or one node).

## Files

| File | Role |
|------|------|
| `property_tests.py` | All property tests and graph generators in one module. |
| `requirements.txt` | Dependencies (includes `numpy` and `scipy` for node classification). |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest property_tests.py -v
```

## Author

Abhishek Malik
SR No: 13-19-01-19-52-25-1-26268

## References

- [NetworkX algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html)
- [Hypothesis](https://hypothesis.readthedocs.io/)
