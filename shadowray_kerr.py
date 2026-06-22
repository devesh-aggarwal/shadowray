"""
shadowray_kerr — V2: Kerr (spinning) black-hole shadow + lensing

V1 (shadowray.py) used spherical symmetry: a photon's fate depended only on its
impact parameter b, so one 1-D table served every pixel. **Spin breaks that.**
Frame dragging makes the deflection depend on the full 2-D image position, so we
must integrate a genuine geodesic per ray. The payoff is the famous asymmetric
shadow -- a flattened "D" -- that the Event Horizon Telescope is built to see.

Two ideas keep this correct and fast instead of a debugging swamp:

  1. Integrate the geodesic in its **second-order form** in Mino time lambda:
         d2r/dlambda^2 = R'(r)/2,    d2theta/dlambda^2 = Theta'(theta)/2
     Unlike dr/dlambda = +/-sqrt(R), this is smooth THROUGH turning points --
     no sign-flip bookkeeping at periapsis, the classic Kerr-tracer foot-gun.

  2. **Vectorize one hand-rolled RK4 over all pixels at once**, freezing each ray
     when it hits the horizon (-> shadow) or returns to the observer (-> sky).
     No per-pixel ODE solver; the whole image marches together as numpy arrays.

Conserved quantities (energy normalised to E = 1): xi = L/E (axial angular
momentum) and eta = Q/E^2 (Carter constant). A screen pixel (alpha, beta) maps to
them by the Bardeen relations. The radial / angular potentials are

    R(r)     = (r^2 + a^2 - a*xi)^2 - Delta * (eta + (xi - a)^2)
    Theta(t) = eta + a^2 cos^2(t) - xi^2 cot^2(t)            Delta = r^2 - 2Mr + a^2

Units: G = c = M = 1, and 0 <= a < 1 (a = M is the extremal limit).

Run this file: it prints the physics checks and writes two PNGs --
kerr_gallery.png (shadow vs. spin) and kerr_hero.png (the money shot).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 1.0
B_CRIT = 3.0 * np.sqrt(3.0) * M           # Schwarzschild shadow radius (a = 0 check)


def horizon_radius(a):
    """Outer event horizon r_+ = M + sqrt(M^2 - a^2)."""
    return M + np.sqrt(M * M - a * a)


# --- Exact analytic shadow outline (Bardeen) --------------------------------

def critical_curve(a, theta_o, n=4000):
    """The shadow's edge as seen at inclination theta_o, exactly.

    Parametrised by the radius r of the unstable spherical photon orbits. Returns
    closed-loop screen coordinates (alpha, beta). For a = 0 it is the circle of
    radius 3*sqrt(3) M; for a > 0 it bulges and flattens into the "D".
    """
    so, co = np.sin(theta_o), np.cos(theta_o)
    if a < 1e-6:
        t = np.linspace(0.0, 2.0 * np.pi, n)
        return B_CRIT * np.cos(t), B_CRIT * np.sin(t)

    # spherical photon orbits span [prograde radius, retrograde radius]
    r_pro = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
    r_ret = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(+a)))
    r = np.linspace(r_pro, r_ret, n)

    xi = (r**2 * (3.0 - r) - a**2 * (1.0 + r)) / (a * (r - 1.0))
    eta = (r**3 * (4.0 * a**2 - r * (r - 3.0) ** 2)) / (a**2 * (r - 1.0) ** 2)
    disc = eta + a**2 * co**2 - xi**2 * (co**2 / so**2)     # = beta^2 on the edge

    keep = disc >= 0.0
    alpha = -xi[keep] / so
    beta = np.sqrt(disc[keep])
    # stitch the +beta branch to the reversed -beta branch into one closed loop
    return (np.concatenate([alpha, alpha[::-1]]),
            np.concatenate([beta, -beta[::-1]]))


# --- Background on the celestial sphere -------------------------------------

def background(theta, phi, spacing=np.pi / 12):
    """Checkerboard on the sphere -> RGB in [0, 1]; its lines bend under lensing."""
    tile = (np.floor(theta / spacing) + np.floor(phi / spacing)) % 2
    dark = np.array([0.05, 0.08, 0.16])
    light = np.array([0.45, 0.62, 0.85])
    return np.where(tile[..., None] > 0.5, light, dark)


# --- The Kerr geodesic kernel (vectorised RK4 in Mino time) -----------------

def _deriv(S, xi, eta, a):
    """Right-hand side of the second-order geodesic system, vectorised.

    State S = [r, pr, theta, ptheta, phi] with pr = dr/dlambda, ptheta = dtheta/dlambda.
    """
    r, pr, th, pth, ph = S
    st, ct = np.sin(th), np.cos(th)
    st = np.where(np.abs(st) < 1e-4, np.copysign(1e-4, st), st)   # keep off the poles
    Delta = r * r - 2.0 * M * r + a * a

    dpr = 2.0 * r * (r * r + a * a - a * xi) - (r - M) * (eta + (xi - a) ** 2)  # R'/2
    dpth = -a * a * st * ct + xi * xi * ct / st**3                              # Theta'/2
    dph = xi / st**2 - a + a * (r * r + a * a - a * xi) / Delta
    return [pr, dpr, pth, dpth, dph]


def _rk4(S, xi, eta, a, h):
    k1 = _deriv(S, xi, eta, a)
    k2 = _deriv([s + 0.5 * h * k for s, k in zip(S, k1)], xi, eta, a)
    k3 = _deriv([s + 0.5 * h * k for s, k in zip(S, k2)], xi, eta, a)
    k4 = _deriv([s + h * k for s, k in zip(S, k3)], xi, eta, a)
    return [s + (h / 6.0) * (a1 + 2.0 * b1 + 2.0 * c1 + d1)
            for s, a1, b1, c1, d1 in zip(S, k1, k2, k3, k4)]


def trace_kerr(a, theta_o, res=300, screen=9.0, r_obs=20.0,
               lam_max=3.0, dlam=0.0025):
    """Backward-trace the whole image plane through the Kerr metric.

    Returns (img, captured_2d, axis, unfinished_frac):
        img            (res, res, 3) RGB in [0, 1]
        captured_2d    bool mask of pixels whose ray fell through the horizon
        axis           the shared alpha/beta coordinate axis (units of M)
        unfinished_frac fraction of rays that neither escaped nor were captured
                        within lam_max (the near-critical photon-ring sliver)
    """
    so, co = np.sin(theta_o), np.cos(theta_o)
    axis = np.linspace(-screen, screen, res)
    A, Bm = np.meshgrid(axis, axis)                 # A = alpha (x), Bm = beta (y)
    alpha, beta = A.ravel(), Bm.ravel()
    N = alpha.size

    # screen pixel -> conserved quantities (Bardeen), and Theta(theta_o) = beta^2
    xi = -alpha * so
    eta = beta**2 + (alpha**2 - a * a) * co**2

    # initial state at the observer, photon heading inward (pr < 0)
    Delta_o = r_obs**2 - 2.0 * M * r_obs + a * a
    R_o = (r_obs**2 + a * a - a * xi) ** 2 - Delta_o * (eta + (xi - a) ** 2)
    S = [np.full(N, r_obs), -np.sqrt(np.maximum(R_o, 0.0)),
         np.full(N, theta_o), beta.copy(), np.zeros(N)]

    rh = horizon_radius(a) + 0.05                   # stop just outside the horizon
    captured = np.zeros(N, bool)
    finished = np.zeros(N, bool)
    th_f, ph_f = np.full(N, theta_o), np.zeros(N)

    idx = np.arange(N)                              # active rays -> pixel indices
    xi_a, eta_a = xi, eta
    for _ in range(int(lam_max / dlam)):
        S = _rk4(S, xi_a, eta_a, a, dlam)
        r, pr = S[0], S[1]
        cap = r <= rh
        esc = (r >= r_obs) & (pr > 0.0)
        done = cap | esc
        if done.any():
            d = idx[done]
            captured[d] = cap[done]
            th_f[d], ph_f[d] = S[2][done], S[4][done]
            finished[d] = True
            keep = ~done
            idx, xi_a, eta_a = idx[keep], xi_a[keep], eta_a[keep]
            S = [s[keep] for s in S]
        if idx.size == 0:
            break

    img = np.zeros((N, 3))
    sky = finished & ~captured                      # escaped to the background
    img[sky] = background(th_f[sky], ph_f[sky])
    # captured rays stay black; near-critical unfinished rays stay black too
    return (img.reshape(res, res, 3), captured.reshape(res, res), axis,
            float(np.mean(~finished)))


# --- Validation: the spinning analogue of V1's 3*sqrt(3) M check ------------

def validate():
    print("validation (Kerr, M = 1):")
    for a in (0.0, 0.5, 0.9, 0.99):
        print(f"  a = {a:.2f}:  horizon r+ = {horizon_radius(a):.3f} M")

    # a = 0 must reproduce the Schwarzschild circle of radius 3*sqrt(3) M
    img, cap, axis, _ = trace_kerr(0.0, np.pi / 2, res=201, screen=7.0)
    A, Bm = np.meshgrid(axis, axis)
    r_shadow = np.sqrt(A[cap] ** 2 + Bm[cap] ** 2).max()
    print(f"  a=0 traced shadow radius = {r_shadow:.4f} M  "
          f"(theory {B_CRIT:.4f} M)")

    # spinning shadows: ray-traced horizontal extent vs. the exact Bardeen curve
    print("  shadow alpha-extent  [traced]      vs [analytic]   (asymmetry = displacement)")
    for a in (0.5, 0.9, 0.99):
        img, cap, axis, _ = trace_kerr(a, np.pi / 2, res=301, screen=9.0)
        A, _Bm = np.meshgrid(axis, axis)
        amin_t, amax_t = A[cap].min(), A[cap].max()
        ca, _cb = critical_curve(a, np.pi / 2)
        amin_a, amax_a = ca.min(), ca.max()
        print(f"    a={a:.2f}:  [{amin_t:+.2f}, {amax_t:+.2f}]   "
              f"vs [{amin_a:+.2f}, {amax_a:+.2f}]   "
              f"center {0.5*(amin_t+amax_t):+.2f} M")


# --- Figures ----------------------------------------------------------------

def _panel(ax, a, theta_o, **kw):
    img, _cap, axis, frac = trace_kerr(a, theta_o, **kw)
    s = axis[-1]
    ax.imshow(img, extent=[-s, s, -s, s], origin="lower")
    cx, cy = critical_curve(a, theta_o)
    ax.plot(cx, cy, color="#ff5a3c", lw=1.1, ls="--", alpha=0.9)   # exact outline
    ax.set_title(f"a = {a:.2f}", color="w", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return frac


def make_gallery():
    """Edge-on shadow as spin grows: the circle slides and flattens into a D."""
    spins = (0.0, 0.5, 0.9, 0.99)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), facecolor="black")
    for ax, a in zip(axes, spins):
        frac = _panel(ax, a, np.pi / 2, res=300, screen=9.0)
        print(f"  gallery a={a:.2f}: unfinished {frac*100:.2f}%")
    fig.suptitle("Kerr black-hole shadow, edge-on  (red = exact Bardeen outline)",
                 color="w", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("kerr_gallery.png", dpi=110, facecolor="black")
    plt.close(fig)
    print("  wrote kerr_gallery.png")


def make_hero():
    """The money shot: near-extremal spin, edge-on, higher resolution."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    frac = _panel(ax, 0.99, np.pi / 2, res=460, screen=9.0)
    ax.set_title("shadowray — Kerr a = 0.99, edge-on", color="w", fontsize=13)
    fig.tight_layout()
    fig.savefig("kerr_hero.png", dpi=120, facecolor="black")
    plt.close(fig)
    print(f"  wrote kerr_hero.png  (unfinished {frac*100:.2f}%)")


def main():
    validate()
    print("render:")
    make_gallery()
    make_hero()


if __name__ == "__main__":
    main()
