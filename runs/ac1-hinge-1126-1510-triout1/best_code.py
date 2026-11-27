"""
Low‑autocorrelation step‑function search.

The optimiser builds a diverse pool of initial windows, then
refines the best candidate with several stages:

1.  L‑BFGS continuation on a log‑sum‑exp smooth max of the
    self‑convolution (coarse optimisation).
2.  Adam optimisation that focuses on the largest convolution
    entries (top‑k).
3.  Exponentiated‑gradient descent on the true max (sub‑gradient)
    with a gradually shrinking active set – an alternative descent
    that often escapes the plateaus left by Adam.
4.  A short greedy hill‑climb to polish the final vector.

All phases respect the global time budget (default 1000 s).  The
returned list is a probability vector (non‑negative, sums to 1);
`evaluate_sequence` rescales it internally, so the official score
equals the fast score 2·n·max(conv) computed during the search.
"""

import time
import numpy as np
from scipy.optimize import minimize

# Optional window imports – fall back to simple definitions if missing.
try:
    from scipy.signal import chebwin, windows
    from scipy.signal.windows import dpss
except Exception:  # pragma: no cover
    chebwin = None
    windows = None
    dpss = None


def next_pow2(x: int) -> int:
    """Smallest power of two ≥ x (x>0)."""
    return 1 << ((x - 1).bit_length())


def softmax(y: np.ndarray) -> np.ndarray:
    """Stable soft‑max (maps ℝⁿ → simplex)."""
    m = np.max(y)
    e = np.exp(y - m)
    return e / e.sum()


def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x ≥ 0, sum(x) = 1}."""
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v, 1.0 / n)
    rho = np.where(cond)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def fast_score(a: np.ndarray, fft_len: int, n: int) -> float:
    """Exact objective 2·n·max(conv) for a probability vector."""
    fa = np.fft.rfft(a, fft_len)
    conv = np.fft.irfft(fa * fa, fft_len)[: 2 * n - 1]
    return float(2.0 * n * conv.max())


def objective_and_grad(y: np.ndarray,
                       n: int,
                       fft_len_conv: int,
                       fft_len_g: int,
                       lam_factor: float = 0.30,
                       edge_weight: float = 0.0) -> tuple[float, np.ndarray]:
    """Smooth max (log‑sum‑exp) of the convolution and its gradient."""
    a = softmax(y)

    # full self‑convolution (length 2·n‑1)
    fa = np.fft.rfft(a, fft_len_conv)
    conv_full = np.fft.irfft(fa * fa, fft_len_conv)[: 2 * n - 1]

    max_c = conv_full.max()
    lam = max_c * lam_factor + 1e-12
    shifted = conv_full - max_c
    exp_shift = np.exp(shifted / lam)
    Z = exp_shift.sum()
    smooth_max = max_c + lam * np.log(Z)
    f = 2.0 * n * smooth_max

    w = exp_shift / Z                     # d smooth_max / d conv_i

    # back‑propagation through convolution: d conv / d a = (rev_a * w)
    a_rev = a[::-1]
    fa_rev = np.fft.rfft(a_rev, fft_len_g)
    fw = np.fft.rfft(w, fft_len_g)
    conv_rev = np.fft.irfft(fa_rev * fw, fft_len_g)

    g_vec = conv_rev[(n - 1):(n - 1 + n)]

    if edge_weight > 0.0:
        pen_grad = np.zeros_like(a)
        pen_grad[0] = 2.0 * edge_weight * a[0]
        pen_grad[-1] = 2.0 * edge_weight * a[-1]
        g_vec = g_vec + pen_grad

    mean_g = np.dot(a, g_vec)
    grad_y = a * (g_vec - mean_g)

    return f, grad_y


# ----------------------------------------------------------------------
# Initialisation helpers (deterministic families)
# ----------------------------------------------------------------------


def spectral_init(n: int) -> np.ndarray:
    """Inverse‑FFT spectral guess, projected onto the simplex."""
    L = next_pow2(2 * n - 1)
    target_len = 2 * n - 1
    uniform = np.full(target_len, 1.0 / target_len)
    u_pad = np.zeros(L)
    u_pad[:target_len] = uniform
    S = np.fft.fft(u_pad)
    amp = np.abs(S)
    phase = np.angle(S)
    sqrt_amp = np.sqrt(np.maximum(amp, 0.0))
    sqrt_phase = phase / 2.0
    A = sqrt_amp * np.exp(1j * sqrt_phase)
    a_complex = np.fft.ifft(A)
    a = np.real(a_complex[:n])
    a[a < 0] = 0.0
    s = a.sum()
    return a / s if s > 0 else np.full(n, 1.0 / n)


def random_initial(n: int, rng: np.random.Generator) -> np.ndarray:
    """Diversified random probability vector."""
    mode = rng.choice(['uniform', 'random', 'exp', 'power',
                       'edge', 'bimodal', 'spike'])
    if mode == 'uniform':
        a = np.full(n, 1.0 / n)
    elif mode == 'random':
        a = rng.random(n)
    elif mode == 'exp':
        rate = rng.uniform(0.001, 0.02)
        a = np.exp(-rate * np.arange(n))
    elif mode == 'power':
        p = rng.uniform(0.2, 2.5)
        a = (np.arange(1, n + 1, dtype=float) ** (-p))
    elif mode == 'edge':
        edge_frac = rng.uniform(0.006, 0.03)
        m_edge = max(1, int(n * edge_frac))
        a = np.full(n, 1.0 / n)
        heavy = np.sqrt(n / 2.0) / n
        a[:m_edge] = heavy
        a[-m_edge:] = heavy
    elif mode == 'bimodal':
        centre = n // 2
        width = rng.integers(int(0.02 * n), int(0.07 * n))
        peak = rng.uniform(0.5, 1.5) / n
        a = np.full(n, 1.0 / n)
        left = max(0, centre - width)
        right = min(n, centre + width)
        a[left:right] = peak
    else:  # spike
        a = np.full(n, 1.0 / n)
        idx = rng.integers(0, n)
        a[idx] = rng.uniform(2.0, 5.0) / n
    a /= a.sum()
    return a


def triangular_init(n: int) -> np.ndarray:
    half = n // 2
    inc = np.linspace(0, 1, half, endpoint=False)
    dec = np.linspace(1, 0, n - half)
    a = np.concatenate([inc, dec])
    a /= a.sum()
    return a


def hann_init(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    a = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))
    a /= a.sum()
    return a


def blackman_init(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    a = (0.42 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1)) +
         0.08 * np.cos(4.0 * np.pi * i / (n - 1)))
    a = np.maximum(a, 0.0)
    a /= a.sum()
    return a


def gaussian_init(n: int, sigma_factor: float = 0.25) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    centre = (n - 1) / 2.0
    sigma = sigma_factor * n
    a = np.exp(-0.5 * ((i - centre) / sigma) ** 2)
    a /= a.sum()
    return a


def kaiser_init(n: int, beta: float = 9.0) -> np.ndarray:
    a = np.kaiser(n, beta)
    a = np.maximum(a, 0.0)
    a /= a.sum()
    return a


def chebwin_init(n: int, att: float = 80.0) -> np.ndarray:
    """Chebyshev window (fallback to Kaiser)."""
    if chebwin is None:
        return kaiser_init(n, beta=8.0)
    a = chebwin(n, at=att)
    a = np.maximum(a, 0.0)
    a /= a.sum()
    return a


def dpss_init(n: int, halfband: float = 0.2) -> np.ndarray:
    """DPSS window (first eigenvector)."""
    if dpss is None:
        return kaiser_init(n, beta=8.0)
    a = dpss(n, halfband, Kmax=1)[0]
    a = np.maximum(a, 0.0)
    a /= a.sum()
    return a


# ----------------------------------------------------------------------
# Optimisation primitives
# ----------------------------------------------------------------------


def lbfgs_continuation(a_start: np.ndarray,
                       lam_schedule,
                       maxiter_per_lam,
                       deadline,
                       fft_len_conv,
                       fft_len_g,
                       n):
    """Run L‑BFGS for each λ in lam_schedule, starting each stage from the previous result."""
    y = np.log(a_start + 1e-12)

    best_a = a_start.copy()
    best_sc = fast_score(best_a, fft_len_conv, n)

    for lam in lam_schedule:
        if time.time() >= deadline:
            break
        try:
            res = minimize(fun=objective_and_grad,
                           x0=y,
                           args=(n, fft_len_conv, fft_len_g, lam, 0.0),
                           method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter_per_lam,
                                    'gtol': 1e-12,
                                    'disp': False})
        except Exception as e:   # pragma: no cover
            print("[lbfgs] exception:", e)
            continue
        y = res.x
        a = softmax(y)
        sc = fast_score(a, fft_len_conv, n)
        if sc < best_sc:
            best_sc = sc
            best_a = a.copy()
    return best_a, best_sc


def adam_topk_optimize(a_start: np.ndarray,
                       deadline: float,
                       n: int,
                       fft_len: int,
                       rng: np.random.Generator,
                       start_top: float = 0.15,
                       end_top: float = 0.001):
    """Adam optimiser that gradually focuses on the true maximum."""
    y = np.log(a_start + 1e-12)
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    beta1_pow = 1.0
    beta2_pow = 1.0
    t = 0

    best_a = softmax(y)
    best_sc = fast_score(best_a, fft_len, n)

    phase_start = time.time()
    total_allowed = max(deadline - phase_start, 1e-9)

    while time.time() < deadline:
        t += 1
        a = softmax(y)

        # full self‑convolution
        fa = np.fft.rfft(a, fft_len)
        conv = np.fft.irfft(fa * fa, fft_len)[: 2 * n - 1]

        # dynamic top‑fraction
        elapsed = time.time() - phase_start
        frac = elapsed / total_allowed
        top_frac = max(end_top, start_top * (1.0 - frac))
        k = max(1, int(np.ceil(top_frac * conv.size)))
        thresh = np.partition(conv, -k)[-k]
        mask = (conv >= thresh).astype(np.float64)

        a_rev = a[::-1]
        fa_rev = np.fft.rfft(a_rev, fft_len)
        fm = np.fft.rfft(mask, fft_len)
        rev_grad = np.fft.irfft(fa_rev * fm, fft_len)

        g_a = 2.0 * rev_grad[(n - 1):(n - 1 + n)]
        grad_a = 2.0 * n * g_a

        mean_g = np.dot(a, grad_a)
        grad_y = a * (grad_a - mean_g)

        m = beta1 * m + (1.0 - beta1) * grad_y
        v = beta2 * v + (1.0 - beta2) * (grad_y ** 2)
        beta1_pow *= beta1
        beta2_pow *= beta2
        m_hat = m / (1.0 - beta1_pow)
        v_hat = v / (1.0 - beta2_pow)

        lr = 0.25 / np.sqrt(t)
        y -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if t % 4000 == 0:
            y += rng.normal(scale=1e-4, size=y.shape)

        a_curr = softmax(y)
        sc = fast_score(a_curr, fft_len, n)
        if sc < best_sc:
            best_sc = sc
            best_a = a_curr.copy()

    return best_a, best_sc


def exponentiated_gradient_optimize(a_start: np.ndarray,
                                   deadline: float,
                                   n: int,
                                   fft_len: int,
                                   rng: np.random.Generator,
                                   start_top: float = 0.20,
                                   end_top: float = 0.001,
                                   eta0: float = 0.8) -> tuple[np.ndarray, float]:
    """
    Exponentiated‑gradient descent on the exact max‑convolution.
    The active set (top‑k entries of the convolution) shrinks linearly
    from `start_top` to `end_top`.  Learning rate η_t = η₀ / √t.
    """
    a = a_start.copy()
    best_a = a.copy()
    best_sc = fast_score(a, fft_len, n)

    start_time = time.time()
    total_allowed = max(deadline - start_time, 1e-9)
    it = 0

    while time.time() < deadline:
        it += 1

        # convolution and current maximum
        fa = np.fft.rfft(a, fft_len)
        conv = np.fft.irfft(fa * fa, fft_len)[: 2 * n - 1]
        max_c = conv.max()

        # shrinking active set
        elapsed = time.time() - start_time
        frac = elapsed / total_allowed
        top_frac = max(end_top, start_top * (1.0 - frac))
        k = max(1, int(np.ceil(top_frac * conv.size)))
        thresh = np.partition(conv, -k)[-k]
        mask = (conv >= thresh).astype(np.float64)
        mask_sum = mask.sum()
        if mask_sum > 0:
            mask = mask / mask_sum   # normalised sub‑gradient weights

        # gradient of the (approximate) max
        a_rev = a[::-1]
        fa_rev = np.fft.rfft(a_rev, fft_len)
        fw = np.fft.rfft(mask, fft_len)
        rev_conv = np.fft.irfft(fa_rev * fw, fft_len)
        g = rev_conv[(n - 1):(n - 1 + n)]

        # exponentiated‑gradient update on the simplex
        eta = eta0 / np.sqrt(it)
        y = np.log(a + 1e-30) - eta * g
        y = y - np.max(y)                 # stabilise exponentiation
        a = np.exp(y)
        a_sum = a.sum()
        if a_sum == 0.0:
            a = np.full_like(a, 1.0 / n)
        else:
            a = a / a_sum

        # occasional jitter to avoid stagnation
        if it % 5000 == 0:
            perturb = rng.normal(scale=1e-5, size=n)
            a = a * np.exp(perturb)
            a = a / a.sum()

        # keep best ever (check every 200 iterations)
        if it % 200 == 0:
            sc = fast_score(a, fft_len, n)
            if sc < best_sc:
                best_sc = sc
                best_a = a.copy()

    return best_a, best_sc


def hill_climb(a: np.ndarray,
               cur_sc: float,
               fft_len: int,
               n: int,
               deadline: float,
               rng: np.random.Generator,
               max_iter: int = 120000,
               delta_frac: float = 0.04):
    """Greedy hill‑climb with tiny mass‑transfer moves."""
    best = a.copy()
    best_sc = cur_sc
    it = 0
    while time.time() < deadline and it < max_iter:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            it += 1
            continue
        delta = best[i] * rng.random() * delta_frac
        if delta <= 0:
            it += 1
            continue
        cand = best.copy()
        cand[i] -= delta
        cand[j] += delta
        sc = fast_score(cand, fft_len, n)
        if sc < best_sc:
            best = cand
            best_sc = sc
        it += 1
    return best, best_sc


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def propose_candidate(seed: int = 42,
                     budget_s: int = 1000,
                     **kwargs):
    """
    Search for a non‑negative step function that minimises the supplied
    evaluation function.  The routine respects the given time budget
    and returns the best sequence found (as a Python list).
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Problem size – a power of two gives fast FFTs.
    # ------------------------------------------------------------------
    n = 8192
    fft_len = next_pow2(2 * n - 1)

    # ------------------------------------------------------------------
    # Build a diverse initial pool (including any previous result in the
    # global variable `height_sequence_1` if present).
    # ------------------------------------------------------------------
    candidates = []

    if 'height_sequence_1' in globals():
        prev = np.array(globals()['height_sequence_1'], dtype=float)
        if prev.ndim == 1 and prev.size > 0:
            x_old = np.linspace(0, 1, prev.size)
            x_new = np.linspace(0, 1, n)
            a0 = np.interp(x_new, x_old, prev)
            a0 = np.clip(a0, 0.0, None)
            s = a0.sum()
            a0 = a0 / s if s > 0 else np.full(n, 1.0 / n)
            candidates.append(a0)

    # deterministic families
    candidates.append(spectral_init(n))
    candidates.append(triangular_init(n))
    candidates.append(hann_init(n))
    candidates.append(blackman_init(n))
    candidates.append(gaussian_init(n))
    candidates.append(kaiser_init(n, beta=9.0))
    candidates.append(chebwin_init(n, att=80.0))
    candidates.append(dpss_init(n, halfband=0.22))

    # random families
    for _ in range(12):
        candidates.append(random_initial(n, rng))

    # pick the best by cheap fast_score
    best_a = candidates[0].copy()
    best_sc = fast_score(best_a, fft_len, n)
    for cand in candidates[1:]:
        sc = fast_score(cand, fft_len, n)
        if sc < best_sc:
            best_sc = sc
            best_a = cand.copy()
    print(f"[init] n={n}, best fast_score={best_sc:.9f}")

    # ------------------------------------------------------------------
    # Time‑budget handling
    # ------------------------------------------------------------------
    start_time = time.time()
    deadline = start_time + budget_s - 0.5   # small safety margin

    # ------------------------------------------------------------------
    # Phase 1 – L‑BFGS continuation (≈30 % of time)
    # ------------------------------------------------------------------
    phase1_deadline = start_time + 0.30 * budget_s
    lam_schedule = [
        0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01,
        0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4,
        5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6,
        5e-7, 2e-7, 1e-7
    ]
    maxiter_per_lam = 2500

    while time.time() < phase1_deadline:
        rand_vec = random_initial(n, rng)
        alpha = rng.random() * 0.30
        a_start = (1.0 - alpha) * best_a + alpha * rand_vec
        a_start /= a_start.sum()
        a_opt, sc_opt = lbfgs_continuation(
            a_start,
            lam_schedule,
            maxiter_per_lam,
            phase1_deadline,
            fft_len,
            fft_len,
            n
        )
        if sc_opt < best_sc:
            best_sc = sc_opt
            best_a = a_opt.copy()
            print(f"[L‑BFGS] new best fast_score={best_sc:.9f}")

    # ------------------------------------------------------------------
    # Phase 2 – Adam with top‑k focus (≈25 % of remaining time)
    # ------------------------------------------------------------------
    phase2_deadline = start_time + 0.55 * budget_s
    if time.time() < phase2_deadline:
        print("[Adam‑topk] start")
        best_a, best_sc = adam_topk_optimize(
            best_a,
            phase2_deadline,
            n,
            fft_len,
            rng,
            start_top=0.12,
            end_top=0.0008
        )
        print(f"[Adam‑topk] best fast_score={best_sc:.9f}")

    # ------------------------------------------------------------------
    # Phase 3 – Exponentiated‑gradient refinement (≈30 % of remaining time)
    # ------------------------------------------------------------------
    phase3_deadline = start_time + 0.85 * budget_s
    if time.time() < phase3_deadline:
        print("[EG] start")
        best_a, best_sc = exponentiated_gradient_optimize(
            best_a,
            phase3_deadline,
            n,
            fft_len,
            rng,
            start_top=0.15,
            end_top=0.0005,
            eta0=0.9
        )
        print(f"[EG] best fast_score={best_sc:.9f}")

    # ------------------------------------------------------------------
    # Phase 4 – final greedy hill‑climb (use any leftover seconds)
    # ------------------------------------------------------------------
    if time.time() < deadline:
        remaining = deadline - time.time()
        hill_deadline = time.time() + min(remaining, 12.0)
        best_a, best_sc = hill_climb(
            best_a,
            best_sc,
            fft_len,
            n,
            hill_deadline,
            rng,
            max_iter=150000,
            delta_frac=0.04
        )
        print(f"[hill‑climb] best fast_score={best_sc:.9f}")

    # ------------------------------------------------------------------
    # Return the best sequence (evaluate_sequence rescales internally)
    # ------------------------------------------------------------------
    final_seq = best_a.tolist()
    final_eval = evaluate_sequence(final_seq)   # official score
    print(f"[final] evaluated score = {final_eval:.9f}")
    # expose the best sequence for possible warm‑starts in later runs
    globals()['height_sequence_1'] = final_seq
    return final_seq