"""Task instructions for the Erdős minimum overlap problem."""


TASK_PROMPT = '''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 2k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {budget_s:.0f}s for your code to run
- **CPUs**: {eval_cpus} available

## Rules
- Define `run(seed=42, budget_s={budget_s:.0f}, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- It is called with NO arguments, so your default for `budget_s` is the one that
  runs. Respect it: track elapsed time and return your best solution before it
  expires, rather than being killed with nothing to show
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- When `initial_h_values` is available, use it directly. Never redefine it or
  copy its entries into the response. Construct all large arrays
  algorithmically; do not print long numeric literals or unrolled tables
- Your function must complete within budget_s seconds and return the best solution found
- The extracted block must define `run` at top level. The harness prepends the
  pre-imported helpers and optional parent construction before execution; do
  not copy those inputs into the block to make it standalone

**Lower is better**. Current record: C₅ ≤ 0.38085. Our goal is to find a construction that shows C₅ ≤ 0.38085.

{state_ctx}
{construction_section}
{code_section}
{direct_output_section}

'''


PARENT_CONSTRUCTION = """
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={size} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""


IMPROVE_PROGRAM = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.\x20
Unless you make a meaningful improvement, you will not be rewarded.'''


WRITE_PROGRAM = '''Write code to optimize this construction.'''


STRATEGY_NOTES = '''
Erdos-specific implementation constraint: `initial_h_values`, when a current
construction is available, is already supplied to the executed program as a
NumPy array. Never plan to copy, serialize, reconstruct, or hard-code that
array. Never propose a long explicit numeric table or an unrolled pattern.
Describe compact algorithmic construction using NumPy operations, formulas,
loops, interpolation, or seeded random generation. The eventual complete
program must fit comfortably in a few hundred lines.

Erdos mathematical checks for the final plan:
- The authoritative objective is max(np.correlate(h, 1-h, mode="full")) *
  (2.0 / n_points), over all 2*n_points-1 lags. This is linear correlation
  with finite support, not circular correlation. Include both lag directions.
- At a nonzero lag, the overlap covers only part of the interval. The sum of
  h over that overlap is not generally its full integral of 1. Do not replace
  the objective with 1 minus a minimum autocorrelation without a derivation
  that preserves these boundary terms and exactly matches the evaluator.
- Products of entries of h make this a generally nonconvex optimization
  problem. A direct CVXPY call is not a solution unless the subproblem you
  formulate satisfies its supported rules. For a convex subproblem or
  relaxation, specify its derivation, candidate recovery, and exact-objective
  acceptance check. Do not claim a global optimum from a local method.
- State how every returned vector satisfies both 0 <= h[i] <= 1 and
  sum(h) = n_points / 2. Clipping and rescaling in succession do not generally
  enforce both constraints; give a valid joint feasibility mechanism.
- Derive dimensions from runtime inputs. If changing resolution, describe how
  to transfer the parent, restore feasibility, and compare exact scores.
  Retain a validated parent as a fallback when one is supplied.'''


COMPACT_SOURCE_CONTRACT = '''## Mandatory compact-source contract

- `initial_h_values` is runtime input when a parent construction exists.
  Reference it directly; NEVER assign to it, paste its values, serialize it,
  or reconstruct it as a literal. If no parent construction was announced,
  create a starting vector under a different name.
- Do not emit any long numeric literal, lookup table, repeated-value list, or
  unrolled pattern. A list/tuple/array literal may contain at most 32 scalar
  values. Build every larger vector algorithmically with NumPy, a compact
  formula, interpolation, tiling, or a loop.
- Keep the complete source concise: at most 900 lines and about 50,000
  characters. Spend tokens on a working algorithm, not comments or embedded
  data. Finish `run`, return the required tuple, and close the single Python
  fence well before the generation limit.'''


DIRECT_OUTPUT = '''\x20
## Required output

Return exactly one complete fenced Python code block, beginning with
```python and ending with ```, and nothing else. Do not output a strategy,
analysis, prose, notes, example usage, a second code block, or a partial
snippet. Finish the program and close the fence well before the response-token
limit.

''' + COMPACT_SOURCE_CONTRACT + '\n\nThese are hard output constraints.'


STAGED_OUTPUT = COMPACT_SOURCE_CONTRACT + '''

These are hard output constraints even if the proposed strategy suggests
otherwise.'''
