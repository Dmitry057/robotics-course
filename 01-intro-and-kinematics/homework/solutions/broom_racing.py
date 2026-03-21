from typing import Callable

import numpy as np

from lib.broom_racing import Configuration, XYZConfiguration, KAPPA_MAX, PHI_MIN, PHI_MAX

# ---------------------------------------------------------------------------
# 2D Dubins path solver (formulas from AndrewWalker/Dubins-Curves)
# ---------------------------------------------------------------------------

def _mod2pi(a):
    return a % (2 * np.pi)


def _dubins2d(x0, y0, th0, xf, yf, thf, rho):
    """
    Shortest 2D Dubins path.
    Returns (segments, total_length) where segments is list of (arc_length, turn_sign).
    turn_sign: +1=left, -1=right, 0=straight.
    """
    dx, dy = xf - x0, yf - y0
    D = np.hypot(dx, dy)
    d = D / rho
    theta = np.arctan2(dy, dx)
    alpha = _mod2pi(th0 - theta)
    beta = _mod2pi(thf - theta)

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cab = np.cos(alpha - beta)  # key difference from previous buggy code

    best_cost = np.inf
    best_params = None

    def _try(t, p, q, types):
        nonlocal best_cost, best_params
        if t < -1e-10 or p < -1e-10 or q < -1e-10:
            return
        t, p, q = max(t, 0.0), max(p, 0.0), max(q, 0.0)
        cost = t + p + q
        if cost < best_cost:
            best_cost = cost
            best_params = (t, p, q, types)

    # LSL
    p_sq = 2 + d*d - 2*cab + 2*d*(sa - sb)
    if p_sq >= 0:
        tmp1 = np.arctan2(cb - ca, d + sa - sb)
        _try(_mod2pi(-alpha + tmp1), np.sqrt(p_sq), _mod2pi(beta - tmp1), (1, 0, 1))

    # RSR
    p_sq = 2 + d*d - 2*cab - 2*d*(sa - sb)
    if p_sq >= 0:
        tmp1 = np.arctan2(ca - cb, d - sa + sb)
        _try(_mod2pi(alpha - tmp1), np.sqrt(p_sq), _mod2pi(-beta + tmp1), (-1, 0, -1))

    # LSR
    p_sq = -2 + d*d + 2*cab + 2*d*(sa + sb)
    if p_sq >= 0:
        p = np.sqrt(p_sq)
        tmp2 = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        _try(_mod2pi(-alpha + tmp2), p, _mod2pi(-_mod2pi(beta) + tmp2), (1, 0, -1))

    # RSL
    p_sq = -2 + d*d + 2*cab - 2*d*(sa + sb)
    if p_sq >= 0:
        p = np.sqrt(p_sq)
        tmp2 = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        _try(_mod2pi(alpha - tmp2), p, _mod2pi(beta - tmp2), (-1, 0, 1))

    # RLR
    tmp = (6.0 - d*d + 2*cab + 2*d*(sa - sb)) / 8.0
    if abs(tmp) <= 1.0:
        p = _mod2pi(2*np.pi - np.arccos(tmp))
        t = _mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb) + _mod2pi(p/2.0))
        q = _mod2pi(alpha - beta - t + _mod2pi(p))
        _try(t, p, q, (-1, 1, -1))

    # LRL
    tmp = (6.0 - d*d + 2*cab + 2*d*(sb - sa)) / 8.0
    if abs(tmp) <= 1.0:
        p = _mod2pi(2*np.pi - np.arccos(tmp))
        t = _mod2pi(-alpha - np.arctan2(ca - cb, d + sa - sb) + p/2.0)
        q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
        _try(t, p, q, (1, -1, 1))

    if best_params is None:
        return None, np.inf

    t, p, q, types = best_params
    segments = [
        (t * rho, types[0]),
        (p * rho, types[1]),
        (q * rho, types[2]),
    ]
    return segments, best_cost * rho


def _dubins2d_sample(x0, y0, th0, segments, rho, t_query):
    """Sample 2D Dubins path at arc-length t_query. Returns (x, y, theta)."""
    x, y, th = x0, y0, th0
    remaining = t_query

    for seg_len, turn in segments:
        if remaining <= 1e-15:
            break
        dl = min(remaining, seg_len)
        remaining -= seg_len
        if dl < 1e-15:
            continue
        if turn == 0:
            x += dl * np.cos(th)
            y += dl * np.sin(th)
        else:
            omega = float(turn) / rho
            dth = omega * dl
            x += (np.sin(th + dth) - np.sin(th)) / omega
            y += -(np.cos(th + dth) - np.cos(th)) / omega
            th += dth

    return x, y, th


# ---------------------------------------------------------------------------
# 3D Dubins via Vana's decoupled approach
# ---------------------------------------------------------------------------

def _check_vertical_feasibility(start_z, start_phi, v_segs, rho_v, v_len, n=300):
    """Check that pitch (=heading in vertical Dubins) stays within [PHI_MIN, PHI_MAX]."""
    for t in np.linspace(0, v_len, n):
        _, _, gamma = _dubins2d_sample(0.0, start_z, start_phi, v_segs, rho_v, t)
        if gamma < PHI_MIN - 0.005 or gamma > PHI_MAX + 0.005:
            return False
    return True


def _solve_dubins3d(start: Configuration, goal: Configuration):
    """
    3D Dubins path: decouple into horizontal (XY) and vertical (S,Z) 2D Dubins paths.

    Horizontal Dubins in XY with radius rho_h.
    Vertical Dubins in (S, Z) plane with radius rho_v.
    Constraint: 1/rho_h^2 + 1/rho_v^2 <= kappa_max^2 = 1.

    The 3D arc-length = vertical arc-length.
    At 3D time t: sample vertical -> (S, Z, gamma), sample horizontal at S -> (x, y, theta).
    EOM satisfied: dx/dt = cos(theta)*cos(gamma), etc.
    """
    rho_min = 1.0 / KAPPA_MAX  # 1.0

    # Algorithm 2 from Vana: start at rho_h = rho_min, double until feasible,
    # then hill-climb optimize.

    # Phase 1: find initial feasible rho_h
    rho_h = rho_min
    feasible_rho_h = None
    for _ in range(30):
        rho_v = _get_rho_v(rho_h, rho_min)
        h_segs, h_len = _dubins2d(start.x, start.y, start.theta,
                                   goal.x, goal.y, goal.theta, rho_h)
        if h_segs is None:
            rho_h *= 2
            continue
        v_segs, v_len = _dubins2d(0.0, start.z, start.phi,
                                   h_len, goal.z, goal.phi, rho_v)
        if v_segs is None:
            rho_h *= 2
            continue
        if _check_vertical_feasibility(start.z, start.phi, v_segs, rho_v, v_len):
            feasible_rho_h = rho_h
            break
        rho_h *= 2

    if feasible_rho_h is None:
        raise ValueError("No feasible 3D Dubins path found")

    # Phase 2: hill-climbing optimization of rho_h
    rho_h = feasible_rho_h
    delta = 0.1 * rho_min
    delta_min = 1e-4

    best_rho_h = rho_h
    best_v_len = _eval_rho_h(start, goal, rho_h, rho_min)

    while abs(delta) > delta_min:
        rho_h_new = max(rho_min, best_rho_h + delta)
        v_len_new = _eval_rho_h(start, goal, rho_h_new, rho_min)
        if v_len_new is not None and v_len_new < best_v_len:
            best_rho_h = rho_h_new
            best_v_len = v_len_new
            delta *= 2
        else:
            delta *= -0.1

    # Build final curve with best rho_h
    rho_h = best_rho_h
    rho_v = _get_rho_v(rho_h, rho_min)
    h_segs, h_len = _dubins2d(start.x, start.y, start.theta,
                               goal.x, goal.y, goal.theta, rho_h)
    v_segs, v_len = _dubins2d(0.0, start.z, start.phi,
                               h_len, goal.z, goal.phi, rho_v)

    total_length = v_len
    sx, sy, sth, sz, sphi = start.x, start.y, start.theta, start.z, start.phi

    def curve(s_param):
        s_val = float(np.atleast_1d(s_param)[0])
        t = s_val * total_length

        # Vertical Dubins at arc-length t -> (S_horiz, Z, gamma=phi)
        S_horiz, Z, gamma = _dubins2d_sample(0.0, sz, sphi, v_segs, rho_v, t)
        S_horiz = np.clip(S_horiz, 0.0, h_len)

        # Horizontal Dubins at horizontal distance S_horiz -> (x, y, theta)
        x, y, theta_h = _dubins2d_sample(sx, sy, sth, h_segs, rho_h, S_horiz)

        return Configuration(float(x), float(y), float(Z), float(theta_h), float(gamma))

    return curve, total_length


def _get_rho_v(rho_h, rho_min):
    inv_sq = 1.0 / rho_min**2 - 1.0 / rho_h**2
    if inv_sq <= 1e-12:
        return 1e6
    return 1.0 / np.sqrt(inv_sq)


def _eval_rho_h(start, goal, rho_h, rho_min):
    """Evaluate vertical path length for given rho_h. Returns None if infeasible."""
    rho_v = _get_rho_v(rho_h, rho_min)
    h_segs, h_len = _dubins2d(start.x, start.y, start.theta,
                               goal.x, goal.y, goal.theta, rho_h)
    if h_segs is None:
        return None
    v_segs, v_len = _dubins2d(0.0, start.z, start.phi,
                               h_len, goal.z, goal.phi, rho_v)
    if v_segs is None:
        return None
    if not _check_vertical_feasibility(start.z, start.phi, v_segs, rho_v, v_len, n=100):
        return None
    return v_len


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gate_pass(
    start: Configuration,
    goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    curve_fn, _ = _solve_dubins3d(start, goal)
    return curve_fn


def catch_snitch(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> Callable[[np.ndarray], Configuration]:
    best_curve = None
    best_length = np.inf

    for theta_f in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        for phi_f in np.linspace(PHI_MIN, PHI_MAX, 7):
            goal = Configuration(goal_xyz.x, goal_xyz.y, goal_xyz.z, theta_f, phi_f)
            try:
                curve_fn, length = _solve_dubins3d(start, goal)
                if length < best_length:
                    best_length = length
                    best_curve = curve_fn
            except ValueError:
                continue

    if best_curve is None:
        raise ValueError("Could not find feasible path to snitch")
    return best_curve


def catch_ball_and_gate(
    start: Configuration,
    intermediate_goal_xyz: XYZConfiguration,
    final_goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    best_combined = None
    best_length = np.inf

    for theta_mid in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        for phi_mid in np.linspace(PHI_MIN, PHI_MAX, 5):
            mid = Configuration(
                intermediate_goal_xyz.x, intermediate_goal_xyz.y,
                intermediate_goal_xyz.z, theta_mid, phi_mid
            )
            try:
                c1, l1 = _solve_dubins3d(start, mid)
                c2, l2 = _solve_dubins3d(mid, final_goal)
                total = l1 + l2
                if total < best_length:
                    best_length = total
                    frac1 = l1 / total

                    def combined(s_param, _c1=c1, _c2=c2, _f1=frac1):
                        s_val = float(np.atleast_1d(s_param)[0])
                        if s_val <= _f1:
                            return _c1(np.array([s_val / _f1 if _f1 > 0 else 0.0]))
                        else:
                            return _c2(np.array([(s_val - _f1) / (1.0 - _f1)]))

                    best_combined = combined
            except ValueError:
                continue

    if best_combined is None:
        raise ValueError("Could not find feasible path for catch_ball_and_gate")
    return best_combined
