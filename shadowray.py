"""
shadowray — Schwarzschild black-hole shadow ray tracer (MVP)

We trace photons *backward* from a virtual image plane to either the black
hole's horizon (-> shadow) or a background grid (-> gravitationally lensed
image). Running this file prints the physics checks and writes shadow.png:
the black shadow disk, a photon-ring glow at its edge, and a lensed grid.

The architectural bet that makes this fast: **spherical symmetry.** In
Schwarzschild spacetime a photon's fate depends ONLY on its impact parameter
b, never on direction. So instead of solving one geodesic per pixel (~100k+
ODE solves), we:

    1. integrate a 1D table ONCE:   b -> (captured?, total swept angle)
    2. render every pixel as a cheap radial lookup by its image-plane radius.

The expensive physics collapses to ~900 integrations; the render is a few
vectorized numpy ops.

Units: geometric units with G = c = 1 and M = 1, so every length is in units
of the mass M (e.g. "5.196 M").

Reading order, following the pipeline:
    integrate_ray()          one geodesic, with capture/escape/periapsis events
    build_deflection_table() sweep b -> the 1-D lookup table (the expensive bit)
    validate()               exact analytic checks (run before trusting a picture)
    background() / render()   image-plane -> sky mapping, then sample + draw
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")          # headless: write a PNG without needing a display
import matplotlib.pyplot as plt


# --- Theory anchors (these gate the physics AND become the test suite) -------

M = 1.0                              # black-hole mass (geometric units)
R_HORIZON = 2.0 * M                  # event horizon, r = 2M
R_PHOTON_SPHERE = 3.0 * M            # unstable photon orbit, r = 3M
B_CRIT = 3.0 * np.sqrt(3.0) * M      # critical impact param = shadow radius, ~5.196 M


# --- The geodesic kernel: one ray, then the 1-D table ------------------------

def integrate_ray(b, r_obs=30.0, phi_max=100.0):
    """Trace ONE photon backward from the observer at impact parameter b.

    We integrate the null-orbit equation in u = 1/r with phi as the
    independent variable:

        du/dphi  = w
        dw/dphi  = 3 M u^2 - u          (i.e. d2u/dphi2 + u = 3 M u^2)

    starting at the observer (u0 = 1/r_obs) and heading inward. The initial
    radial slope is fixed by the impact parameter via the null condition
        (du/dphi)^2 = 1/b^2 - u^2 (1 - 2 M u).

    Three solve_ivp events decide the photon's fate:
        capture   u reaches the horizon 1/(2M)                 -> falls in
        escape    u returns to u0 on the far side              -> reaches sky
        periapsis w = 0 (closest approach), recorded for tests

    Returns (captured, phi_total, r_peri):
        captured    True if the photon hit the horizon
        phi_total   total azimuth swept from observer to escape (radians); this
                    is the angle to where the ray lands on the sky. NaN if
                    captured. The gravitational bending is phi_total minus the
                    flat-space value 2*arccos(b/r_obs) -- see validate().
        r_peri      smallest radius reached (closest approach), in units of M
    """
    u0 = 1.0 / r_obs
    # null condition sets |du/dphi| at the observer; +sign = heading inward
    w0 = np.sqrt(max(1.0 / b**2 - u0**2 * (1.0 - 2.0 * M * u0), 0.0))

    def rhs(phi, y):
        u, w = y
        return (w, 3.0 * M * u * u - u)

    def hit_horizon(phi, y):
        return y[0] - 1.0 / (2.0 * M)
    hit_horizon.terminal = True
    hit_horizon.direction = +1.0           # u rising through the horizon

    def back_to_obs(phi, y):
        return y[0] - u0
    back_to_obs.terminal = True
    back_to_obs.direction = -1.0           # u falling back to the observer radius

    def periapsis(phi, y):
        return y[1]                        # w = du/dphi = 0 at closest approach
    periapsis.direction = -1.0             # turning from inward to outward

    sol = solve_ivp(rhs, (0.0, phi_max), (u0, w0),
                    events=(hit_horizon, back_to_obs, periapsis),
                    rtol=1e-9, atol=1e-12, max_step=0.5)

    if sol.t_events[0].size > 0:                       # capture
        return True, np.nan, R_HORIZON

    phi_total = sol.t_events[1][0] if sol.t_events[1].size > 0 else sol.t[-1]
    if sol.t_events[2].size > 0:
        r_peri = 1.0 / sol.y_events[2][0][0]
    else:
        r_peri = 1.0 / np.max(sol.y[0])
    return False, float(phi_total), float(r_peri)


def build_deflection_table(b_max=14.0, n_samples=900, r_obs=30.0):
    """Map impact parameter b -> (captured?, total swept angle), integrated once.

    Returns three aligned 1-D arrays:
        b_grid      sampled impact parameters, ~0 .. b_max
        captured    bool, True if the photon spirals into the hole
        phi_total   total azimuth swept to the sky (radians) for escaping photons

    This is the package's one expensive step: ~n_samples geodesic solves. Every
    pixel is then a cheap radial lookup into this table (see render()).
    """
    b_grid = np.linspace(1e-3, b_max, n_samples)       # avoid b=0 (1/b blows up)
    captured = np.zeros(n_samples, dtype=bool)
    phi_total = np.full(n_samples, np.nan)
    for i, b in enumerate(b_grid):
        captured[i], phi_total[i], _ = integrate_ray(b, r_obs=r_obs)

    # Captured rays (and the few that wind without resolving) have no finite
    # swept angle. Clamp them to the strongest escaping value so the lookup stays
    # finite -- those pixels land inside the shadow and are painted black anyway.
    escaping = ~captured & np.isfinite(phi_total)
    if escaping.any():
        phi_total[~escaping] = phi_total[escaping].max()
    return b_grid, captured, phi_total


def find_capture_boundary(r_obs=30.0, lo=1.0, hi=10.0, iters=40):
    """Bisect for the impact parameter dividing capture from escape.

    `lo` must be captured and `hi` must escape. Converges on b_crit = 3*sqrt(3) M.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        captured, _, _ = integrate_ray(mid, r_obs=r_obs)
        lo, hi = (mid, hi) if captured else (lo, mid)
    return 0.5 * (lo + hi)


def measured_shadow_radius(b_grid, captured):
    """Largest impact parameter that still falls in = the shadow radius.

    This is the validation hook: from the integrated table it lands on
    B_CRIT = 3*sqrt(3) M.
    """
    if not captured.any():
        return 0.0
    return float(b_grid[captured].max())


# --- Step 5: procedural background on the celestial sphere -------------------

def background(theta, phi, grid_spacing=np.pi / 14):
    """Checkerboard on the sphere -> RGB array in [0, 1].

    theta (polar) and phi (azimuth) may be arrays of any shape; the returned
    array has a trailing length-3 (RGB) axis. Straight grid lines on the sphere
    visibly bend once we look at them through the lens -- that bending is the
    whole point of the demo.
    """
    tile = (np.floor(theta / grid_spacing) + np.floor(phi / grid_spacing)) % 2
    dark = np.array([0.05, 0.08, 0.16])        # deep navy
    light = np.array([0.45, 0.62, 0.85])       # periwinkle
    return np.where(tile[..., None] > 0.5, light, dark)


# --- Step 4: map the image plane through the lens to the sky -----------------

def render(resolution=600, r_obs=30.0, b_max=10.0, grid_spacing=np.pi / 14):
    """Trace the whole image plane and return an (H, W, 3) RGB array in [0, 1].

    Pipeline (this is the architecture the whole package hangs on):
        1. lay down an image-plane grid in units of M
        2. per pixel: impact parameter rho and azimuth psi
        3. look up capture + total swept angle from the 1-D table by rho
        4. turn (rho, psi, swept-angle) into a direction on the sky
        5. sample the background there; paint captured pixels black; glow the ring

    Geometry of step 4: by spherical symmetry each photon stays in the plane
    containing the black hole, the observer (on +z), and the image azimuth psi.
    In that plane the observer sits at angle 0 and the escaping ray points at
    angle phi_total. Rotating that back into 3-D, the sky direction is

        d = (sin(phi_total) cos(psi), sin(phi_total) sin(psi), cos(phi_total))

    which we read off as sky (theta, phi). For a far observer with no hole this
    gives phi_total -> pi, i.e. d -> -z (straight through) -- the flat-space limit.
    """
    # 1. image-plane grid, spanning [-b_max, b_max] M in each axis
    axis = np.linspace(-b_max, b_max, resolution)
    x, y = np.meshgrid(axis, axis)
    rho = np.hypot(x, y)               # impact parameter of each pixel
    psi = np.arctan2(y, x)             # azimuth around the optical axis

    # 2-3. one table for the whole image (cover the corners, rho up to b_max*sqrt2)
    b_grid, captured, phi_table = build_deflection_table(
        b_max=b_max * np.sqrt(2.0), r_obs=r_obs)
    phi_total = np.interp(rho, b_grid, phi_table)         # swept angle per pixel
    b_shadow = measured_shadow_radius(b_grid, captured)
    shadow = rho < b_shadow

    # 4. sky direction d from the swept angle, then read off sky (theta, phi)
    s = np.sin(phi_total)
    dx, dy, dz = s * np.cos(psi), s * np.sin(psi), np.cos(phi_total)
    theta = np.arccos(np.clip(dz, -1.0, 1.0))
    phi = np.arctan2(dy, dx)

    # 5. sample the lensed background
    img = background(theta, phi, grid_spacing)

    # 6 (tuning): a warm photon-ring glow where rays pile up at b_crit
    glow = np.exp(-((rho - b_shadow) / 0.30) ** 2)
    img = img + 0.6 * glow[..., None] * np.array([1.0, 0.78, 0.45])

    # carve out the black shadow disk last, so the ring rims its edge
    img[shadow] = 0.0
    return np.clip(img, 0.0, 1.0)


def validate():
    """Cheap physics gate -- run this BEFORE trusting any picture.

    A wrong image looks plausibly wrong a hundred ways; these analytic anchors
    are exact and catch a broken kernel immediately.
    """
    print("validation (Schwarzschild, M = 1):")

    # 1. capture boundary / shadow radius == 3*sqrt(3) M
    b_c = find_capture_boundary(r_obs=30.0)
    print(f"  capture boundary    b = {b_c:.4f} M   "
          f"(theory {B_CRIT:.4f} M,  err {abs(b_c - B_CRIT):.1e})")

    # 2. photon sphere: a near-critical escaping ray grazes r = 3M
    _, _, r_peri = integrate_ray(B_CRIT + 1e-4, r_obs=30.0)
    print(f"  near-crit periapsis r = {r_peri:.4f} M   "
          f"(theory {R_PHOTON_SPHERE:.4f} M)")

    # 3. weak field: deflection -> 4M/b as b grows (needs r_obs >> b).
    #    deflection = (total swept angle) - (flat-space swept angle).
    r_far = 2000.0
    for b in (20.0, 50.0, 100.0):
        _, phi_total, _ = integrate_ray(b, r_obs=r_far)
        defl = phi_total - 2.0 * np.arccos(b / r_far)
        print(f"  weak field b={b:5.0f} M  defl = {defl:.5f} rad   "
              f"(4M/b = {4.0 * M / b:.5f})")


def main():
    validate()

    print("render:")
    img = render()
    out = "shadow.png"
    plt.imsave(out, np.clip(img, 0.0, 1.0), origin="lower")
    print(f"  wrote {out}  ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
