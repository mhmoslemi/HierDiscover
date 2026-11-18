"""Shared instructions for the autocorrelation tasks."""


STRATEGY_NOTES = '''
Autocorrelation-specific scientific and execution requirements:
- Write the final plan as normal, readable English prose with ordinary spaces
  and line breaks. Never emit serialized tokenizer surfaces, escaped token
  streams, tool/backend error messages, execution traces, or mixed-language
  fragments. Before closing the block, confirm that all six requested sections
  are present in order and readable independently of the reasoning transcript.
- If a current construction exists, `height_sequence_1` is already supplied
  to the executed program as a one-dimensional NumPy array. Never copy,
  serialize, reconstruct, or hard-code that array. Read it directly as a
  possible initialization; this does not require retaining its resolution
  or construction family.
- The entrypoint must return a non-empty Python `list` of finite real numbers.
  Keep the returned coefficients explicitly nonnegative and no larger than
  1000; do not depend on the evaluator's defensive clipping to repair a search
  candidate. Retain a separately copied, exactly re-evaluated best candidate
  and a concrete valid fallback throughout the run.
- The authoritative evaluator uses full *linear* self-convolution, producing
  2*n-1 coefficients. It is not circular convolution. Derive every surrogate,
  gradient, or convolution index against that definition. Verify updates to
  the saved best candidate and the final return with `evaluate_sequence`.
  Internal search iterates may worsen or use surrogates; do not mistake their
  surrogate scores for verified improvements.
- Positive rescaling below the coefficient cap leaves the objective unchanged.
  Do not claim that changing scale alone is an improvement, and do not let
  normalization interact with clipping in a way that changes the candidate.
- Products of the submitted coefficients make the search generally nonconvex.
  A direct CVXPY call is not a solution unless the stated subproblem satisfies
  the solver's rules. Explain any relaxation or alternating subproblem, how a
  candidate is recovered, and how it is checked by the original evaluator.
- Choose algorithms for the actual problem size and execution resources.
  Account for arrays, temporary copies, solver workspace, and evaluation cost.
  Dense operators, finite differences, and LP methods are options when their
  costs fit; for larger searches, consider structured, sparse, matrix-free,
  FFT, analytic-gradient, or reduced-subproblem approaches as appropriate.
  Explain the chosen formulation's scaling, rather than assuming a library
  call makes a large allocation or solve affordable.
- Schedule expensive evaluations and solver calls using elapsed time inside
  the generated program. Surrogates may guide exploration, but leave time for
  exact validation and returning the saved best valid candidate. Do not impose
  an arbitrary number of evaluations independent of the actual runtime.
- Sequence length and construction family are search choices, not fixed by
  the parent, seed program, or literature example. Derive dimensions from the
  candidate being evaluated. If changing resolution, explain the construction
  or resampling rule, normalization, validity checks, and exact comparison.
  Explain initialization when no parent is available. The coefficient cap of
  1000 is a value bound, not a limit on the number of coefficients.'''


CONSTRUCTION_LENGTH = "\nLength of the construction: {size}"
