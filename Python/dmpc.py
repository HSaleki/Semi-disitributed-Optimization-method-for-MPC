"""
Distributed active-set QP solver.

This is the Python port of the MATLAB implementation. The problem is

    min  sum_i  1/2 z_i' Q_i z_i + p_i' z_i
    s.t. G_i z_i = g_i
         H_i z_i <= h_i
         sum_i A_i z_i <= b

Local nodes form reduced KKT systems. The central node handles the coupled
constraints through the sum of the local Schur blocks. The active set is
followed with a homotopy parameter sigma, starting at 1 and ending at 0.

Requires numpy and scipy.
"""

import argparse

import numpy as np
from scipy.linalg import ldl, null_space, qr, solve_discrete_are

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny



def ninf(v):
    """Infinity norm helper."""
    v = np.asarray(v)
    return float(np.abs(v).max()) if v.size else 0.0


# ---------------------------------------------------------------- factorization

def seq_qr_rankreveal(A, tol=None):
    """Householder QR used for the active constraint matrix.

    Dependent columns are skipped in their original order.
    """
    n, m = A.shape
    if tol is None:
        tol = np.sqrt(EPS) * np.linalg.norm(A, 1) if A.size else np.sqrt(EPS)

    V = np.zeros((n, m))
    beta = np.zeros(m)
    R = np.zeros((n, m))
    kept = np.zeros(m, dtype=bool)
    k = 0

    for i in range(m):
        a = A[:, i].copy()
        for j in range(k):
            v = V[:, j]
            a -= beta[j] * (v @ a) * v

        if np.linalg.norm(a[k:]) <= tol:
            continue                      # dependent column -> drop the constraint

        kept[i] = True
        x = a[k:]
        s = x[0] if x[0] != 0.0 else 1.0
        alpha = -np.sign(s) * np.linalg.norm(x)
        v = x.copy()
        v[0] -= alpha
        v /= np.linalg.norm(v)

        V[k:, k] = v
        beta[k] = 2.0
        R[:k, k] = a[:k]
        R[k, k] = alpha
        k += 1

    return V[:, :k], beta[:k], R[:k, :k], kept, k


class LDLSolve:
    """Factorization of the reduced Hessian."""

    def __init__(self, M):
        self.n = M.shape[0]
        if self.n:
            self.lu, self.d, _ = ldl(M)

    def solve(self, rhs):
        if self.n == 0:
            return np.zeros_like(rhs)
        y = np.linalg.solve(self.lu, rhs)
        y = np.linalg.solve(self.d, y)
        return np.linalg.solve(self.lu.T, y)


# --------------------------------------------------------------------- logging

class Logger:
    """Counts the number of array elements exchanged."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.central = {"send": 0, "recv": 0}
        self.local = {}

    def _bump(self, d, tag, payload):
        if payload is None:
            return
        d[tag] += int(np.size(payload))

    def central_log(self, payload, tag):
        self._bump(self.central, tag, payload)

    def local_log(self, node_id, payload, tag):
        d = self.local.setdefault(node_id, {"send": 0, "recv": 0})
        self._bump(d, tag, payload)

    def totals(self):
        c = self.central["send"] + self.central["recv"]
        l = {k: v["send"] + v["recv"] for k, v in self.local.items()}
        return c, l


# ------------------------------------------------------------------ local node

class LocalNode:
    def __init__(self, node_id, Q, p, G, g, H, h, A, x0,
                 loc_dual, loc_active, coup_dual, coup_active, logger):
        self.id = node_id
        self.log = logger

        self.B = np.asarray(Q, float)
        self.p = np.asarray(p, float).ravel()
        self.G = np.asarray(G, float)
        self.g = np.asarray(g, float).ravel()
        self.H = np.asarray(H, float)
        self.h = np.asarray(h, float).ravel()
        self.A = np.asarray(A, float)

        self.nx = self.B.shape[0]
        self.neq = self.G.shape[0]
        self.nineq = self.H.shape[0]
        self.ncoup = self.A.shape[0]

        self.Zg = null_space(self.G) if self.neq else np.eye(self.nx)
        self.Hz = self.H @ self.Zg
        self.GGt = self.G @ self.G.T if self.neq else None

        self.x = np.asarray(x0, float).ravel().copy()
        loc_dual = np.asarray(loc_dual, float).ravel()
        self.lam = loc_dual[:self.neq].copy()
        self.mu = loc_dual[self.neq:].copy()
        self.dlam = np.zeros(self.neq)
        self.dmu = np.zeros(self.nineq)

        self.act_l = np.asarray(loc_active, bool).copy()
        self.mu[~self.act_l] = 0.0
        self.nu = np.asarray(coup_dual, float).ravel().copy()
        self.dnu = np.zeros(self.ncoup)
        self.act_c = np.asarray(coup_active, bool).copy()
        self.nu[~self.act_c] = 0.0

        self.flag = False
        self.sigma = 1.0
        self.tau = 0.0
        self.dtau = 0.0
        self.change_l = False
        self.change_c = False

        # Keep nearly active rows slightly away from the boundary.
        self.slack_rel = 1e-6

        self.S_i = None
        self.rho_i = None
        self.dS_i = None
        self.drho_i = None

        self._initialize()

    # -- setup -------------------------------------------------------------

    def _row_scale(self, mask, Hx):
        return np.maximum.reduce([np.abs(self.h[mask]), np.abs(Hx[mask]),
                                  np.ones(int(mask.sum()))])

    def _initialize(self):
        g0 = self.G @ self.x if self.neq else np.zeros(0)
        Hx = self.H @ self.x

        h0 = self.h + np.maximum(Hx - self.h, 0.0)
        h0[self.act_l] = Hx[self.act_l]
        self.mu[~self.act_l] = 0.0

        too = (~self.act_l) & (Hx - h0 > -1e-10)
        if too.any():
            h0[too] = Hx[too] + self.slack_rel * self._row_scale(too, Hx)

        self.GHA = np.hstack([self.G.T, self.H[self.act_l].T])
        self.V, self.beta, self.T, kept, self.r = seq_qr_rankreveal(self.GHA)

        if not kept.all():
            dropped = self._resolve_dependency()
            self.GHA = np.hstack([self.G.T, self.H[self.act_l].T])
            self.V, self.beta, self.T, kept2, self.r = seq_qr_rankreveal(self.GHA)
            if not kept2.all():
                raise RuntimeError(f"node {self.id}: rank deficient after drop")
            if len(dropped):
                m = np.zeros(self.nineq, bool)
                m[dropped] = True
                h0[dropped] = Hx[dropped] + self.slack_rel * self._row_scale(m, Hx)

        self.p_0 = -(self.B @ self.x + self.G.T @ self.lam
                     + self.H.T @ self.mu + self.A.T @ self.nu)
        self.d_p = self.p_0 - self.p
        self.d_g = g0 - self.g
        self.d_h = h0 - self.h

        self.p_tau, self.g_tau, self.h_tau = self.p_0.copy(), g0.copy(), h0.copy()
        self.flag = False
        self.r_p = self.d_p
        self.r_g = -self.d_g

        self._factor_local()
        if self.act_c.any():
            self._coupled_blocks(force=True)
        else:
            self._close_uncoupled()

    def _factor_local(self):
        """Build the reduced system and compute the uncoupled step."""
        Qi = np.eye(self.nx)
        for j in range(self.r - 1, -1, -1):
            v = self.V[j:, j]
            Qi[j:, :] -= self.beta[j] * np.outer(v, v @ Qi[j:, :])
        self.Qi = Qi
        self.Y = Qi[:, :self.r]
        self.Z = Qi[:, self.r:]

        self.YBY = self.Y.T @ self.B @ self.Y
        self.ZBY = self.Z.T @ self.B @ self.Y
        self.ZBZ = self.Z.T @ self.B @ self.Z

        self.Yr_p = self.Y.T @ self.r_p
        Zr_p = self.Z.T @ self.r_p

        r_gh = np.concatenate([self.r_g, -self.d_h[self.act_l]]) \
            if self.act_l.any() else self.r_g
        self.s_Y = np.linalg.solve(self.T.T, r_gh) if self.r else np.zeros(0)

        self.YBYs = self.YBY @ self.s_Y
        self.red = LDLSolve(self.ZBZ)
        self.ZBYs = self.ZBY @ self.s_Y
        self.s_Zbar = self.red.solve(Zr_p - self.ZBYs)
        self.dxbar = self.Qi @ np.concatenate([self.s_Y, self.s_Zbar])

    def _coupled_blocks(self, force=False):
        Ac = self.A[self.act_c]
        if force or self.change_c or self.change_l:
            self.ZAT = self.Z.T @ Ac.T
        self.Mi = self.red.solve(self.ZAT)
        S = self.ZAT.T @ self.Mi
        rho = Ac @ self.dxbar
        if force or self.change_c:
            self.S_i, self.rho_i = S, rho
        elif self.change_l:
            self.dS_i = S - self.S_i
            self.drho_i = rho - self.rho_i
            self.S_i, self.rho_i = S, rho

    def _close_uncoupled(self):
        """No coupled constraints are active."""
        self.s_Z = self.s_Zbar
        self.dx = self.dxbar
        self.YBZs = self.ZBY.T @ self.s_Z
        self._recover_multipliers(self.Yr_p - self.YBYs - self.YBZs)
        self.homotopy_step()

    def _recover_multipliers(self, rhs):
        dlm = np.linalg.solve(self.T, rhs) if self.r else np.zeros(0)
        self.dlam = dlm[:self.neq]
        self.dmu = np.zeros(self.nineq)
        self.dmu[self.act_l] = dlm[self.neq:]

    def _assemble(self):
        if self.change_l:
            self.GHA = np.hstack([self.G.T, self.H[self.act_l].T])
            self.V, self.beta, self.T, kept, self.r = seq_qr_rankreveal(self.GHA)

            if self.r < min(self.GHA.shape):
                if (np.flatnonzero(~kept) < self.neq).any():
                    raise RuntimeError(f"node {self.id}: equality block rank deficient")
                dropped = self._resolve_dependency()
                self.dmu[dropped] = 0.0
                self.GHA = np.hstack([self.G.T, self.H[self.act_l].T])
                self.V, self.beta, self.T, _, self.r = seq_qr_rankreveal(self.GHA)
                if self.r < min(self.GHA.shape):
                    raise RuntimeError(f"node {self.id}: still rank deficient")
            self._factor_local()

        if self.act_c.any():
            self._coupled_blocks()
        else:
            self._close_uncoupled()

    # -- dependency handling -----------------------------------------------

    @staticmethod
    def _dual_ratio(muW, c):
        idx = np.flatnonzero(c > 0)
        if idx.size == 0:
            return np.inf, -1
        ratios = muW[idx] / c[idx]
        k = int(np.argmin(ratios))
        return ratios[k], idx[k]

    def _resolve_dependency(self):
        """Drop dependent active rows. The multiplier change is transferred to the
        equality multipliers so stationarity is preserved.
        """
        mu0, lam0 = self.mu.copy(), self.lam.copy()
        tol = 1e-10
        dropped = []

        nrm = np.linalg.norm(self.Hz, axis=1)
        dead = self.act_l & (nrm <= tol * max(nrm.max(initial=0.0), TINY))
        if dead.any():
            self.mu[dead] = 0.0
            self.act_l[dead] = False
            dropped += list(np.flatnonzero(dead))

        while True:
            W = np.flatnonzero(self.act_l)
            if W.size <= 1:
                break
            Hw = self.Hz[W]
            s = np.maximum(np.linalg.norm(Hw, axis=1), TINY)
            D = null_space((Hw / s[:, None]).T, rcond=tol)
            if D.size == 0:
                break

            c = D[:, 0] / s
            tp, jp = self._dual_ratio(self.mu[W], c)
            tm, jm = self._dual_ratio(self.mu[W], -c)
            if tp <= tm:
                t, j = tp, jp
            else:
                t, j, c = tm, jm, -c
            if not np.isfinite(t):
                break

            muW = self.mu[W] - t * c
            muW[j] = 0.0
            self.mu[W] = np.maximum(muW, 0.0)
            self.act_l[W[j]] = False
            dropped.append(W[j])

        if dropped and self.neq:
            r = self.H.T @ (self.mu - mu0)
            self.lam = self.lam - np.linalg.solve(self.GGt, self.G @ r)
            jump = np.linalg.norm(self.H.T @ (self.mu - mu0)
                                  + self.G.T @ (self.lam - lam0), np.inf)
            if jump > 1e-8 * max(np.linalg.norm(self.H.T @ mu0, np.inf), 1.0):
                raise RuntimeError(f"node {self.id}: drop left a stationarity jump {jump:.3e}")

        return np.array(dropped, dtype=int)

    # -- messaging ---------------------------------------------------------

    def send_initial(self):
        Ax = self.A @ self.x
        self.log.local_log(self.id, Ax, "send")
        if self.act_c.any():
            self.log.local_log(self.id, self.S_i, "send")
            self.log.local_log(self.id, self.rho_i, "send")
            return Ax, self.S_i, self.rho_i
        return Ax, None, None

    def send_to_central(self):
        self.log.local_log(self.id, self.S_i, "send")
        self.log.local_log(self.id, self.rho_i, "send")
        return self.S_i, self.rho_i, self.flag

    def send_change(self):
        self.log.local_log(self.id, self.dS_i, "send")
        self.log.local_log(self.id, self.drho_i, "send")
        return self.dS_i, self.drho_i, self.flag

    def send_step(self):
        Adx = self.A @ self.dx
        self.log.local_log(self.id, Adx, "send")
        self.log.local_log(self.id, self.dtau, "send")
        return Adx, self.dtau

    def receive_delta_nu(self, dnu_active):
        self.log.local_log(self.id, dnu_active, "recv")
        self.dnu[:] = 0.0
        self.dnu[self.act_c] = dnu_active

        self.s_Z = self.s_Zbar - self.Mi @ dnu_active
        self.dx = self.Qi @ np.concatenate([self.s_Y, self.s_Z])
        self.YBZs = self.ZBY.T @ self.s_Z
        self.YAT = self.Y.T @ self.A[self.act_c].T

        self._recover_multipliers(self.Yr_p - self.YBYs - self.YBZs
                                  - self.YAT @ dnu_active)
        self.homotopy_step()

    def drop_coupled(self, dropped):
        self.act_c[dropped] = False
        self.nu[dropped] = 0.0
        self.dnu[dropped] = 0.0
        self.change_c = True
        self._assemble()

    # -- ratio tests -------------------------------------------------------

    def homotopy_step(self):
        mu_s = max(ninf(self.mu[self.act_l])
                   if self.act_l.any() else 0.0, 1.0)
        dmu_s = max(ninf(self.dmu[self.act_l])
                    if self.act_l.any() else 0.0, TINY)

        snap = self.act_l & (np.abs(self.mu) <= 4 * EPS * mu_s)
        self.mu[snap] = 0.0

        neg = (self.dmu < -1e-10 * dmu_s) & self.act_l
        if not neg.any():
            self.dtau_d, self.k_d = np.inf, None
        else:
            idx = np.flatnonzero(neg)
            ratios = np.maximum(-self.mu[idx] / self.dmu[idx], 0.0)
            k = int(np.argmin(ratios))
            self.dtau_d, self.k_d = ratios[k], idx[k]

        cand = ~self.act_l
        ci = np.flatnonzero(cand)
        Hdx = self.H[cand] @ self.dx
        num = np.maximum(self.h_tau[cand] - self.H[cand] @ self.x, 0.0)
        den = Hdx + self.d_h[cand]
        den_s = max(ninf(Hdx)
                    + ninf(self.d_h),
                    ninf(self.h), 1.0)

        ok = den > 1e-10 * den_s
        if not ok.any():
            self.dtau_p, self.k_p = np.inf, None
        else:
            ratios = num[ok] / den[ok]
            k = int(np.argmin(ratios))
            self.dtau_p, self.k_p = ratios[k], ci[np.flatnonzero(ok)[k]]

        self.dtau = min(self.dtau_p, self.dtau_d)
        assert self.dtau >= 0.0, f"node {self.id}: negative step"

    # -- stepping ----------------------------------------------------------

    def _apply_reflectors(self, a, forward=True):
        a = a.copy()
        rng = range(self.r) if forward else range(self.r - 1, -1, -1)
        for j in rng:
            v = self.V[j:, j]
            a[j:] -= self.beta[j] * v * (v @ a[j:])
        return a

    def receive_delta_tau(self, dtau, flag, act_c_new):
        if flag == 1:
            self.change_l = True
            self.change_c = bool(np.any(self.act_c != act_c_new))
        elif flag == 0:
            self.change_l = False
            self.change_c = bool(np.any(self.act_c != act_c_new))

        if flag is None:                                   # terminal signal
            self.dtau = min(dtau, self.sigma)
            self._advance()
            mu_s = max(np.linalg.norm(self.mu, np.inf), 1.0)
            self.mu[self.act_l & (np.abs(self.mu) <= 4 * EPS * mu_s)] = 0.0
            self.mu[~self.act_l] = 0.0
            if (self.mu < -1e-10 * mu_s).any():
                raise RuntimeError(
                    f"node {self.id}: terminal step drove mu to {self.mu.min():.3e}; "
                    "the path was closed past a dual blocking point")
            self.mu[self.mu < 0] = 0.0
            return

        self.log.local_log(self.id, dtau, "recv")
        self.log.local_log(self.id, flag, "recv")
        self.log.local_log(self.id, act_c_new, "recv")

        self.dtau = min(dtau, self.sigma)
        self.act_c = act_c_new.copy()
        self._advance()

        mu_s = max(np.linalg.norm(self.mu, np.inf), 1.0)
        self.mu[self.act_l & (np.abs(self.mu) <= 4 * EPS * mu_s)] = 0.0
        self.mu[~self.act_l] = 0.0

        if flag == 1:
            # A primal and dual event can occur at the same step.
            did_p = (self.dtau == self.dtau_p) and self.k_p is not None
            did_d = (self.dtau == self.dtau_d) and self.k_d is not None
            if did_p:
                self._activate(self.k_p)
            if did_d:
                if not did_p:
                    self._check_curvature(self.k_d)
                self.act_l[self.k_d] = False
                self.mu[self.k_d] = 0.0

        self.mu[~self.act_l] = 0.0
        mu_s = max(np.linalg.norm(self.mu, np.inf), 1.0)
        if (self.mu < -1e-10 * mu_s).any():
            raise RuntimeError(f"node {self.id}: negative multiplier {self.mu.min():.3e}")
        self.mu[self.mu < 0] = 0.0

        self._update_tau()

    def _advance(self):
        self.x = self.x + self.dtau * self.dx
        self.lam = self.lam + self.dtau * self.dlam
        self.mu = self.mu + self.dtau * self.dmu
        self.nu = self.nu + self.dtau * self.dnu
        self.sigma -= self.dtau
        self.tau = 1.0 - self.sigma

    def _update_tau(self):
        # Update the homotopy data.
        self.p_tau = self.p + self.sigma * self.d_p
        self.g_tau = self.g + self.sigma * self.d_g
        self.h_tau = self.h + self.sigma * self.d_h
        self.flag = self.sigma < 1e-14
        self._assemble()

    def _activate(self, k):
        """Add row k, exchanging an active row if necessary."""
        QTH = self._apply_reflectors(self.H[k].copy(), forward=True)
        YH, ZH = QTH[:self.r], QTH[self.r:]

        p_y = np.zeros(self.r)
        p_z = self.red.solve(ZH)
        p_ = self._apply_reflectors(np.concatenate([p_y, p_z]), forward=False)

        rhs = YH - self.YBY @ p_y - self.ZBY.T @ p_z
        zx = np.linalg.solve(self.T, rhs) if self.r else np.zeros(0)
        zeta = zx[:self.neq]
        xi = np.zeros(self.nineq)
        xi[self.act_l] = zx[self.neq:]

        tol_dep = np.sqrt(EPS) * max(np.linalg.norm(self.GHA, 1),
                                     np.linalg.norm(self.H[k], 1))
        if np.linalg.norm(ZH) <= tol_dep:
            ind = (xi > 0) & self.act_l
            if not ind.any():
                raise RuntimeError(f"node {self.id}: QP appears infeasible at row {k}")
            cand = np.flatnonzero(ind)
            ratios = self.mu[ind] / xi[ind]
            j = int(np.argmin(ratios))
            theta = max(ratios[j], 0.0)
            leaving = cand[j]

            self.lam -= theta * zeta
            self.mu -= theta * xi
            self.act_l[leaving] = False
            self.mu[leaving] = 0.0
            self.mu[k] = theta

        self.act_l[k] = True

    def _check_curvature(self, k):
        """Check the zero-curvature case after releasing a constraint."""
        W = np.flatnonzero(self.act_l)
        e_A = np.zeros(W.size)
        e_A[W == k] = 1.0

        p_y = np.linalg.solve(self.T.T, np.concatenate([np.zeros(self.neq), -e_A])) \
            if self.r else np.zeros(0)
        p_z = self.red.solve(-self.ZBY @ p_y)
        p_ = self._apply_reflectors(np.concatenate([p_y, p_z]), forward=False)

        zx = np.linalg.solve(self.T, -self.YBY @ p_y - self.ZBY.T @ p_z) \
            if self.r else np.zeros(0)
        if np.linalg.norm(zx, np.inf) >= 1e-14 * max(np.linalg.norm(self.mu, np.inf), 1.0):
            return

        h_tau = self.h + self.sigma * self.d_h
        free = ~self.act_l
        num = h_tau[free] - self.H[free] @ self.x
        den = self.H[free] @ p_
        den_s = max(ninf(den), TINY)
        ok = den > 1e-10 * den_s
        if not ok.any():
            raise RuntimeError(f"node {self.id}: unbounded along the released direction")

        ratios = np.maximum(num[ok] / den[ok], 0.0)
        j = int(np.argmin(ratios))
        if ratios[j] > 1e10:
            raise RuntimeError(f"node {self.id}: QP appears infeasible")
        self.x = self.x + ratios[j] * p_
        self.act_l[np.flatnonzero(free)[np.flatnonzero(ok)[j]]] = True

    # -- accessors ---------------------------------------------------------

    @property
    def dual(self):
        mu = self.mu.copy()
        mu[~self.act_l] = 0.0
        mu[mu < 0] = 0.0
        return np.concatenate([self.lam, mu])

    def residuals(self):
        rL = np.linalg.norm(self.B @ self.x + self.p + self.G.T @ self.lam
                            + self.H.T @ self.mu + self.A.T @ self.nu, np.inf)
        r_eq = np.linalg.norm(self.G @ self.x - self.g, np.inf) if self.neq else 0.0
        sc_h = max(1.0, np.linalg.norm(self.h, np.inf))
        sc_m = max(1.0, np.linalg.norm(self.mu, np.inf))
        r_tight = ninf(self.H[self.act_l] @ self.x - self.h[self.act_l]) / sc_h
        r_viol = max(0.0, np.max(self.H @ self.x - self.h)) / sc_h
        r_comp = np.linalg.norm(self.mu * (self.h - self.H @ self.x), np.inf) / (sc_h * sc_m)
        return np.array([rL, r_eq, r_tight, r_viol, r_comp])


# ---------------------------------------------------------------- central node

class CentralNode:
    def __init__(self, nodes, b, coup_dual, coup_active, logger):
        self.nodes = nodes
        self.log = logger
        self.b = np.asarray(b, float).ravel()
        self.n = self.b.size

        self.nu = np.asarray(coup_dual, float).ravel().copy()
        self.dnu = np.zeros(self.n)
        self.act_c = np.asarray(coup_active, bool).copy()

        self.sigma = 1.0
        self.tau = 0.0
        self.dtau = 0.0
        self.exit_flag = False
        self.dtaus = np.zeros(len(nodes) + 1)
        self.stopflag = np.zeros(len(nodes))
        self.flagc = False
        self.change_c = bool(self.act_c.any())
        self.id = 0
        self.n_zero = 0

        self.Ax_sum = np.zeros(self.n)
        self.Adx_sum = np.zeros(self.n)
        nA = int(self.act_c.sum())
        self.S_sum = np.zeros((nA, nA))
        self.rho_sum = np.zeros(nA)

        self.slack_rel = 1e-6
        self.sigma_tol = 1e-14
        # Prevent an endless sequence of zero-length steps.
        self.zero_budget = 2 * (self.n + sum(nd.nineq for nd in nodes))

        self._initialize()

    def _initialize(self):
        for nd in self.nodes:
            Ax, S, rho = nd.send_initial()
            self.log.central_log(Ax, "recv")
            self.Ax_sum += Ax
            if S is not None:
                self.log.central_log(S, "recv")
                self.log.central_log(rho, "recv")
                self.S_sum += S
                self.rho_sum += rho

        viol = self.Ax_sum - self.b
        sc = np.maximum(np.abs(viol), 1.0)
        nu_s = max(np.linalg.norm(self.nu, np.inf), 1.0)
        purge = self.act_c & (self.nu <= 1e-12 * nu_s) & (viol < -1e-10 * sc)
        if purge.any():
            self.act_c[purge] = False
            self.nu[purge] = 0.0

        self.b_0 = self.b + np.maximum(viol, 0.0)
        self.b_0[self.act_c] = self.Ax_sum[self.act_c]
        self.nu[~self.act_c] = 0.0

        res = self.Ax_sum - self.b_0
        too = (~self.act_c) & (res > -1e-10)
        if too.any():
            s = np.maximum.reduce([np.abs(self.b[too]), np.abs(self.Ax_sum[too]),
                                   np.ones(int(too.sum()))])
            self.b_0[too] = self.Ax_sum[too] + self.slack_rel * s

        self.r_b = self.b - self.b_0
        self.b_tau = self.b_0.copy()
        self.flagc = False

    def _update_tau(self):
        self.b_tau = self.b - self.sigma * self.r_b     # == b bitwise at sigma == 0
        self.flagc = self.sigma < 1e-14

    def _drop_rows(self, dropped):
        self.act_c[dropped] = False
        self.nu[dropped] = 0.0
        for nd in self.nodes:
            nd.drop_coupled(dropped)
        self.change_c = True
        nA = int(self.act_c.sum())
        self.S_sum = np.zeros((nA, nA))
        self.rho_sum = np.zeros(nA)

    def aggregate_and_solve(self):
        for _ in range(len(self.nodes) + 1):
            if not self.act_c.any():
                self.dnu[:] = 0.0
                break

            if self.change_c:
                nA = int(self.act_c.sum())
                self.S_sum = np.zeros((nA, nA))
                self.rho_sum = np.zeros(nA)
                for i, nd in enumerate(self.nodes):
                    S, rho, sflag = nd.send_to_central()
                    self.log.central_log(S, "recv")
                    self.log.central_log(rho, "recv")
                    self.S_sum += S
                    self.rho_sum += rho
                    self.stopflag[i] = sflag
            else:
                dS, drho, sflag = self.nodes[self.id].send_change()
                self.log.central_log(dS, "recv")
                self.log.central_log(drho, "recv")
                self.S_sum += dS
                self.rho_sum += drho
                self.stopflag[self.id] = sflag

            _, R, P = qr(self.S_sum, mode="economic", pivoting=True)
            rd = np.abs(np.diag(R))
            tol = max(self.S_sum.shape) * np.spacing(rd[0])
            r = int((rd > tol).sum())

            if r < self.S_sum.shape[1]:
                active = np.flatnonzero(self.act_c)
                self._drop_rows(active[P[r:]])
                continue

            self.dnu_active = np.linalg.solve(self.S_sum,
                                              self.rho_sum - self.r_b[self.act_c])
            self.dnu[:] = 0.0
            self.dnu[self.act_c] = self.dnu_active
            for nd in self.nodes:
                self.log.central_log(self.dnu_active, "send")
                nd.receive_delta_nu(self.dnu_active)
            break

        self.Adx_sum = np.zeros(self.n)
        for i, nd in enumerate(self.nodes):
            Adx, dtau_i = nd.send_step()
            self.log.central_log(Adx, "recv")
            self.log.central_log(dtau_i, "recv")
            self.Adx_sum += Adx
            self.dtaus[i] = dtau_i

    def homotopy_step(self):
        # Dual ratio test.
        dnu_s = max(ninf(self.dnu[self.act_c]), TINY)
        ind_nu = (self.dnu < -1e-10 * dnu_s) & self.act_c
        if not ind_nu.any():
            dtau_d, ka = np.inf, None
        else:
            idx = np.flatnonzero(ind_nu)
            ratios = np.maximum(-self.nu[idx] / self.dnu[idx], 0.0)
            j = int(np.argmin(ratios))
            dtau_d, ka = ratios[j], idx[j]

        # Primal ratio test.
        cand = ~self.act_c
        ci = np.flatnonzero(cand)
        den = self.Adx_sum[cand] - self.r_b[cand]
        num = np.maximum(self.b_tau[cand] - self.Ax_sum[cand], 0.0)
        den_s = max(np.linalg.norm(self.Adx_sum, np.inf)
                    + np.linalg.norm(self.r_b, np.inf), TINY)
        ok = den > 1e-10 * den_s
        if not ok.any():
            dtau_p, l = np.inf, None
        else:
            ratios = num[ok] / den[ok]
            j = int(np.argmin(ratios))
            dtau_p, l = ratios[j], ci[np.flatnonzero(ok)[j]]

        self.dtaus[-1] = min(dtau_p, dtau_d)
        self.id = int(np.argmin(self.dtaus))
        dtau_c = self.dtaus[self.id]
        assert dtau_c >= 0, f"central: negative step from agent {self.id}"
        self.dtau = dtau_c

        if min(self.dtau, self.sigma) <= 0:
            self.n_zero += 1
        else:
            self.n_zero = 0

        if self.sigma <= self.sigma_tol:
            step = self.sigma
        elif self.n_zero >= self.zero_budget:
            raise RuntimeError(
                f"degenerate cycle: {self.n_zero} zero-length steps at "
                f"sigma = {self.sigma:.6e}")
        else:
            step = min(self.dtau, self.sigma)

        assert step <= self.dtau + 1e-12 * max(self.sigma, 1) or self.sigma <= self.sigma_tol

        self.Ax_sum += step * self.Adx_sum
        self.nu += step * self.dnu
        nu_s = np.linalg.norm(self.nu, np.inf)
        self.nu[self.act_c & (np.abs(self.nu) <= 4 * EPS * max(nu_s, 1.0))] = 0.0

        self.sigma -= step
        self.tau = 1.0 - self.sigma

        if self.sigma < 1e-14:
            self.exit_flag = True
            for nd in self.nodes:
                nd.receive_delta_tau(step, None, self.act_c)
            return True

        tol_tie = 1e-12 * max(self.sigma, 1.0)
        tied = np.isfinite(self.dtaus) & (self.dtaus <= dtau_c + tol_tie)
        local_acts = np.flatnonzero(tied[:-1])

        if tied[-1]:
            self.change_c = True
            if dtau_p <= dtau_c + tol_tie and l is not None:
                self.act_c[l] = True
            if dtau_d <= dtau_c + tol_tie and ka is not None:
                self.act_c[ka] = False
                self.nu[ka] = 0.0
        else:
            # Rebuild if several local active sets change.
            self.change_c = local_acts.size > 1

        if self.change_c:
            nA = int(self.act_c.sum())
            self.S_sum = np.zeros((nA, nA))
            self.rho_sum = np.zeros(nA)

        for i, nd in enumerate(self.nodes):
            self.log.central_log(self.dtau, "send")
            self.log.central_log(1 if i in local_acts else 0, "send")
            self.log.central_log(self.act_c, "send")
            nd.receive_delta_tau(self.dtau, 1 if i in local_acts else 0, self.act_c)

        self._update_tau()
        return False

    @property
    def dual(self):
        nu = self.nu.copy()
        nu[~self.act_c] = 0.0
        nu[nu < 0] = 0.0
        return nu

    def residuals(self):
        r = ninf(self.Ax_sum[self.act_c] - self.b[self.act_c])
        return np.array([r, np.abs(self.nu @ (self.b - self.Ax_sum))])


# -------------------------------------------------------------------- driver

def solve(data, max_iters=10000):
    """Solve the coupled QP."""
    M = len(data["Q"])
    logger = data.get("logger") or Logger()

    nodes = [LocalNode(i, data["Q"][i], data["p"][i], data["G"][i], data["g"][i],
                       data["H"][i], data["h"][i], data["A"][i], data["z0"][i],
                       data["loc_dual"][i], data["loc_active"][i],
                       data["coup_dual"], data["coup_active"], logger)
             for i in range(M)]
    central = CentralNode(nodes, data["b"], data["coup_dual"],
                          data["coup_active"], logger)

    hist = {"nA_c": [], "id": [], "sigma": [], "change_c": []}
    done = False
    for it in range(1, max_iters + 1):
        central.aggregate_and_solve()
        done = central.homotopy_step()

        hist["nA_c"].append(int(central.act_c.sum()))
        hist["id"].append(central.id)
        hist["sigma"].append(central.sigma)
        hist["change_c"].append(central.change_c)
        if done:
            break

    out = {
        "status": "optimal" if done else "max_iters",
        "iter": it,
        "sigma": central.sigma,
        "z": [nd.x.copy() for nd in nodes],
        "dual_l": [nd.dual for nd in nodes],
        "dual_c": central.dual,
        "active_l": [nd.act_l.copy() for nd in nodes],
        "active_c": central.act_c.copy(),
        "res": [nd.residuals() for nd in nodes] + [central.residuals()],
        "log": hist,
        "logger": logger,
    }
    if not done:
        out["z"] = None                 # do not hand back a non-converged point
    return out


def kkt_residual(data, sol):
    """Compute a scaled KKT residual."""
    Q = np.block([[data["Q"][i] if i == j else np.zeros((data["Q"][i].shape[0],
                                                         data["Q"][j].shape[1]))
                   for j in range(len(data["Q"]))] for i in range(len(data["Q"]))])
    p = np.concatenate(data["p"])
    G = np.block([[data["G"][i] if i == j else np.zeros((data["G"][i].shape[0],
                                                         data["G"][j].shape[1]))
                   for j in range(len(data["G"]))] for i in range(len(data["G"]))])
    g = np.concatenate(data["g"])
    H = np.block([[data["H"][i] if i == j else np.zeros((data["H"][i].shape[0],
                                                         data["H"][j].shape[1]))
                   for j in range(len(data["H"]))] for i in range(len(data["H"]))])
    h = np.concatenate(data["h"])
    A = np.hstack(data["A"])
    b = data["b"]

    z = np.concatenate(sol["z"])
    ne = data["G"][0].shape[0]
    lam = np.concatenate([d[:ne] for d in sol["dual_l"]])
    mu = np.concatenate([d[ne:] for d in sol["dual_l"]])
    nu = sol["dual_c"]

    y = np.concatenate([mu, nu])
    r = np.concatenate([H @ z - h, A @ z - b])
    s_p = max(1.0, np.linalg.norm(h, np.inf), np.linalg.norm(b, np.inf))
    s_d = max(1.0, np.linalg.norm(y, np.inf))

    return max(
        np.linalg.norm(Q @ z + p + G.T @ lam + H.T @ mu + A.T @ nu, np.inf)
        / max(1.0, np.linalg.norm(p, np.inf)),
        np.linalg.norm(G @ z - g, np.inf) / max(1.0, np.linalg.norm(g, np.inf)),
        max(0.0, r.max()) / s_p,
        max(0.0, -y.min()) / s_d,
        np.linalg.norm(y * r, np.inf) / (s_p * s_d),
    )


# =============================================================== platoon example
#
# z_i = [u0; x1; u1; x2; ... ; u_{N-1}; xN],  x = [p; v; a],  u = jerk.
# Followers track the leader at a fixed reference gap; consecutive vehicles are
# coupled by  p_i(k) - p_{i-1}(k) <= -D_hard.

def chain3rd(Ts):
    A = np.array([[1, Ts, 0.5 * Ts ** 2],
                  [0, 1, Ts],
                  [0, 0, 1.0]])
    B = np.array([Ts ** 3 / 6, Ts ** 2 / 2, Ts])
    return A, B.reshape(3, 1)


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def idx_u(k, nu, nx):
    return slice(k * (nu + nx), k * (nu + nx) + nu)


def idx_x(k, nu, nx):
    return slice(k * (nu + nx) + nu, (k + 1) * (nu + nx))


def build_dynamics(Ad, Bd, N):
    nx, nu = Ad.shape[0], Bd.shape[1]
    Aeq = np.zeros((N * nx, N * (nu + nx)))
    Beq = np.zeros((N * nx, nx))
    for k in range(N):
        rows = slice(k * nx, (k + 1) * nx)
        Aeq[rows, idx_x(k, nu, nx)] = np.eye(nx)
        Aeq[rows, idx_u(k, nu, nx)] = -Bd
        if k >= 1:
            Aeq[rows, idx_x(k - 1, nu, nx)] = -Ad
        else:
            Beq[rows, :] = Ad
    return Aeq, Beq


def build_hessian(Q, R, P, N):
    nx, nu = Q.shape[0], R.shape[0]
    H = np.zeros((N * (nu + nx),) * 2)
    for k in range(N):
        Qk = P if k == N - 1 else Q
        H[idx_u(k, nu, nx), idx_u(k, nu, nx)] = 2 * R
        H[idx_x(k, nu, nx), idx_x(k, nu, nx)] = 2 * Qk
    return H


def build_local_ineq(bounds, N, nx=3, nu=1):
    nZ = N * (nu + nx)
    Eu, Ev, Ea = (np.zeros((N, nZ)) for _ in range(3))
    for k in range(N):
        Eu[k, idx_u(k, nu, nx)] = 1.0
        Ev[k, idx_x(k, nu, nx)] = [0, 1, 0]
        Ea[k, idx_x(k, nu, nx)] = [0, 0, 1]
    Aloc = np.vstack([Eu, -Eu, Ev, -Ev, Ea, -Ea])
    bloc = np.concatenate([
        bounds["u_max"] * np.ones(N), -bounds["u_min"] * np.ones(N),
        bounds["v_max"] * np.ones(N), -bounds["v_min"] * np.ones(N),
        bounds["a_max"] * np.ones(N), -bounds["a_min"] * np.ones(N)])
    return Aloc, bloc


def build_coupling(M, N, D, leader_p, nx=3, nu=1):
    P0 = leader_p[1:]
    Ep = np.zeros((N, N * (nu + nx)))
    for k in range(N):
        Ep[k, idx_x(k, nu, nx)] = [1, 0, 0]

    Acell = [np.zeros((M * N, N * (nu + nx))) for _ in range(M)]
    b = -D * np.ones(M * N)

    Acell[0][:N, :] = Ep
    b[:N] = P0 - D
    for e in range(1, M):
        rows = slice(e * N, (e + 1) * N)
        Acell[e][rows, :] += Ep
        Acell[e - 1][rows, :] -= Ep
    return Acell, b


def build_zref(xref, uref):
    nx, N = xref.shape
    nu = uref.shape[0]
    z = np.zeros(N * (nu + nx))
    for k in range(N):
        z[idx_u(k, nu, nx)] = uref[:, k]
        z[idx_x(k, nu, nx)] = xref[:, k]
    return z


def shift_z(Ad, Bd, Z_prev, leader_p, D_ref, K, u_ref, bounds, R, N, P, nx=3, nu=1):
    """Shift the previous MPC solution and fill the last stage with the LQR law.
    """
    Cv, Ca = np.array([0, 1, 0]), np.array([0, 0, 1])
    Reff = float((R + Bd.T @ P @ Bd).item())
    cc = np.array([1.0, -1.0, Cv @ Bd.ravel(), -(Cv @ Bd.ravel()),
                   Ca @ Bd.ravel(), -(Ca @ Bd.ravel())])
    gx = np.column_stack([np.zeros(nx), np.zeros(nx), Cv, -Cv, Ca, -Ca])

    out, act_N, mu_N, lam_corr = [], [], [], []
    for i, Z in enumerate(Z_prev, start=1):
        Zs = np.zeros_like(Z)
        Zs[:-(nx + nu)] = Z[nx + nu:]

        xNm1 = Z[-nx:]
        xr = np.array([leader_p[-2] - i * D_ref, 20.0, 0.0])
        u_lqr = float(u_ref - K.ravel() @ (xNm1 - xr))

        xfree = Ad @ xNm1
        dd = np.array([bounds["u_max"], -bounds["u_min"],
                       bounds["v_max"] - Cv @ xfree, -bounds["v_min"] + Cv @ xfree,
                       bounds["a_max"] - Ca @ xfree, -bounds["a_min"] + Ca @ xfree])

        iu, il = np.flatnonzero(cc > 0), np.flatnonzero(cc < 0)
        hi_j = iu[np.argmin(dd[iu] / cc[iu])]
        lo_j = il[np.argmax(dd[il] / cc[il])]
        hi, lo = dd[hi_j] / cc[hi_j], dd[lo_j] / cc[lo_j]

        kr = None
        if lo > hi:
            u_pad = min(max(u_lqr, bounds["u_min"]), bounds["u_max"])
        elif u_lqr > hi:
            u_pad, kr = hi, hi_j
        elif u_lqr < lo:
            u_pad, kr = lo, lo_j
        else:
            u_pad = u_lqr

        if kr is None:
            act_N.append(-1)
            mu_N.append(0.0)
            lam_corr.append(np.zeros(nx))
        else:
            act_N.append((kr + 1) * N - 1)          # row index of stage N, block kr
            m = 2 * Reff * (u_lqr - u_pad) / cc[kr]
            mu_N.append(m)
            lam_corr.append(-gx[:, kr] * m)

        Zs[-(nx + nu):] = np.concatenate([[u_pad], (Ad @ xNm1 + Bd.ravel() * u_pad)])
        out.append(Zs)

    return out, np.array(act_N), np.array(mu_N), np.column_stack(lam_corr)


def shift_mask(mask, N):
    Mm = mask.reshape(-1, N).copy()        # blocks are contiguous, N per block
    new = np.zeros_like(Mm)
    new[:, :N - 1] = Mm[:, 1:]
    new[:, N - 1] = Mm[:, N - 1]
    return new.reshape(-1)


def shift_dual_local(dual, N, nx, neq, P, z0, leader_p, D_ref, v_ref, lam_corr):
    out = []
    for i, d in enumerate(dual, start=1):
        xr = np.array([leader_p[-1] - i * D_ref, v_ref, 0.0])
        xN = z0[i - 1][-nx:]

        lam = d[:neq].reshape(N, nx).T.copy()       # column k is stage k+1
        new = np.zeros_like(lam)
        new[:, :N - 1] = lam[:, 1:]
        new[:, N - 1] = -2 * P @ (xN - xr) + lam_corr[:, i - 1]

        mu = shift_mask_num(d[neq:], N)
        out.append(np.concatenate([new.T.reshape(-1), mu]))
    return out


def shift_mask_num(v, N):
    Mm = v.reshape(-1, N).copy()
    new = np.zeros_like(Mm)
    new[:, :N - 1] = Mm[:, 1:]
    new[:, N - 1] = Mm[:, N - 1]
    return new.reshape(-1)


def run_platoon(n_iter=20, N=20, M=5, Ts=1.0, verbose=True):
    D_hard, D_ref = 30.0, 35.0          # D_ref > D_hard, or every coupled row is
    v_ref, u_ref = 20.0, 0.0            # weakly active and the ratio tests degenerate
    nx, nu = 3, 1

    bounds = {"u_min": -2.0, "u_max": 2.0, "v_min": 0.0, "v_max": 30.0,
              "a_min": -3.0, "a_max": 3.0}
    Qs = np.diag([0.3, 0.6, 0.9])
    Rs = np.array([[1.0]])

    Ad, Bd = chain3rd(Ts)
    Aeq, Beq = build_dynamics(Ad, Bd, N)
    K, P = dlqr(Ad, Bd, Qs, Rs)
    Hess = build_hessian(Qs, Rs, P, N)

    leader_p = np.linspace(0.0, N * Ts * v_ref, N + 1)
    Aloc, bloc = build_local_ineq(bounds, N)
    neq, nineq = Aeq.shape[0], Aloc.shape[0]

    seed_ic = ([-60, 18, -1], [-110, 25, 1], [-170, 15, 0],
               [-240, 12, -1], [-280, 5, 1])
    x = [np.array(seed_ic[i], float) if i < len(seed_ic)
         else np.array([-280.0 - 55 * (i - 4), 10.0, 0.0])
         for i in range(M)]

    z0 = [np.zeros(N * (nx + nu)) for _ in range(M)]
    loc_active = [np.zeros(nineq, bool) for _ in range(M)]
    loc_dual = [np.zeros(neq + nineq) for _ in range(M)]
    coup_active = np.zeros(M * N, bool)
    coup_dual = np.zeros(M * N)

    logger = Logger()
    rho_hist, iter_hist, comm_c, comm_l = [], [], [], []
    traj = [np.array(x)]                 # traj[k][i] = state of vehicle i at step k
    lead = [leader_p[0]]
    Z_prev = None

    for t in range(n_iter):
        A_coup, b_coup = build_coupling(M, N, D_hard, leader_p)
        logger.reset()

        grad = []
        for i in range(1, M + 1):
            xref = np.vstack([leader_p[1:] - i * D_ref,
                              v_ref * np.ones(N),
                              np.zeros(N)])
            grad.append(-Hess @ build_zref(xref, np.zeros((nu, N))))

        if Z_prev is not None:
            z0, act_N, mu_N, lam_corr = shift_z(Ad, Bd, Z_prev, leader_p, D_ref,
                                                K, u_ref, bounds, Rs, N, P)
            loc_active = [shift_mask(a, N) for a in loc_active]
            loc_dual = shift_dual_local(loc_dual, N, nx, neq, P, z0,
                                        leader_p, D_ref, v_ref, lam_corr)

            rowsN = np.arange(1, 7) * N - 1
            for i in range(M):
                loc_active[i][rowsN] = False        # shift_z owns the stage-N block
                loc_dual[i][neq + rowsN] = 0.0
                if act_N[i] >= 0:
                    loc_active[i][act_N[i]] = True
                    loc_dual[i][neq + act_N[i]] = mu_N[i]

            coup_active = shift_mask(coup_active, N)
            coup_dual = shift_mask_num(coup_dual, N)

            Ax = sum(A_coup[i] @ z0[i] for i in range(M))
            rowsC = np.arange(1, M + 1) * N - 1
            tol = 10 * EPS * max(np.linalg.norm(Ax, np.inf),
                                 np.linalg.norm(b_coup, np.inf))
            coup_active[rowsC] = (Ax[rowsC] - b_coup[rowsC]) > tol * D_hard
            coup_dual[rowsC] = 0.0

        data = {
            "Q": [Hess] * M, "p": grad,
            "G": [Aeq] * M, "g": [Beq @ x[i] for i in range(M)],
            "H": [Aloc] * M, "h": [bloc] * M,
            "A": A_coup, "b": b_coup, "z0": z0,
            "loc_active": loc_active, "loc_dual": loc_dual,
            "coup_active": coup_active, "coup_dual": coup_dual,
            "logger": logger,
        }

        sol = solve(data)
        if sol["status"] != "optimal":
            raise RuntimeError(f"step {t}: solver returned {sol['status']}")

        rho = kkt_residual(data, sol)
        c_tot, l_tot = logger.totals()
        rho_hist.append(rho)
        iter_hist.append(sol["iter"])
        comm_c.append(c_tot)
        comm_l.append(sum(l_tot.values()))

        if verbose:
            print(f"t={t:3d}  K={sol['iter']:4d}  rho={rho:.3e}  "
                  f"|A_c|={int(sol['active_c'].sum()):3d}  "
                  f"central={c_tot:7d}  local={sum(l_tot.values()):8d}")

        Z_prev = sol["z"]
        loc_active, loc_dual = sol["active_l"], sol["dual_l"]
        coup_active, coup_dual = sol["active_c"], sol["dual_c"]
        x = [Ad @ x[i] + Bd.ravel() * Z_prev[i][0] for i in range(M)]
        leader_p = np.append(leader_p[1:], leader_p[-1] + Ts * v_ref)
        traj.append(np.array(x))
        lead.append(leader_p[0])

    return {"rho": np.array(rho_hist), "iter": np.array(iter_hist),
            "central": np.array(comm_c), "local": np.array(comm_l),
            "traj": np.array(traj), "leader": np.array(lead),
            "Ts": Ts, "D_hard": D_hard, "D_ref": D_ref, "M": M}



def plot_run(out, path=None):
    """Plot the closed-loop results."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, lead, Ts, M = out["traj"], out["leader"], out["Ts"], out["M"]
    nk = X.shape[0]
    t = np.arange(nk) * Ts
    tk = np.arange(1, nk) * Ts
    col = plt.cm.viridis(np.linspace(0, 0.85, M))

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))

    ax[0, 0].plot(t, lead, "k", lw=2, label="leader")
    for i in range(M):
        ax[0, 0].plot(t, X[:, i, 0], color=col[i], label=f"car {i+1}")
    ax[0, 0].set(xlabel="time [s]", ylabel="position [m]", title="positions")
    ax[0, 0].legend(fontsize=8)

    for i in range(M):
        ax[0, 1].plot(t, X[:, i, 1], color=col[i])
    ax[0, 1].axhline(20, ls=":", c="k", lw=1)
    ax[0, 1].set(xlabel="time [s]", ylabel="velocity [m/s]", title="velocities")

    for i in range(M):
        ax[0, 2].plot(t, X[:, i, 2], color=col[i])
    ax[0, 2].set(xlabel="time [s]", ylabel="acceleration [m/s2]",
                 title="accelerations")

    # gap to the vehicle ahead; car 1's predecessor is the leader
    ahead = np.column_stack([lead, X[:, :-1, 0]])
    gaps = ahead - X[:, :, 0]
    for i in range(M):
        ax[1, 0].plot(t, gaps[:, i], color=col[i], label=f"{i}->{i+1}")
    ax[1, 0].axhline(out["D_hard"], ls="--", c="r", lw=1.2, label="D_hard")
    ax[1, 0].axhline(out["D_ref"], ls=":", c="g", lw=1.2, label="D_ref")
    ax[1, 0].set(xlabel="time [s]", ylabel="gap [m]", title="inter-vehicle gaps")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogy(tk, np.maximum(out["rho"], 1e-18), "o-", ms=3)
    ax[1, 1].set(xlabel="time [s]", ylabel="rho", title="KKT residual")

    a = ax[1, 2]
    a.plot(tk, out["central"], label="central")
    a.plot(tk, out["local"], "--", label="local (all nodes)")
    a.set_yscale("log")
    a.set(xlabel="time [s]", ylabel="doubles per step", title="data transferred")
    a2 = a.twinx()
    a2.plot(tk, out["iter"], ":k", lw=1.2)
    a2.set_ylabel("outer iterations K(t)")
    a2.set_yscale("log")
    a.legend(fontsize=8, loc="upper right")

    for row in ax:
        for a_ in row:
            a_.grid(alpha=0.3)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")
    else:
        plt.show()
    return fig


def _random_problem(seed=0, M=3, n=6, ne=2, nc=4):
    rng = np.random.default_rng(seed)
    data = {k: [] for k in ("Q", "p", "G", "g", "H", "h", "A", "z0",
                            "loc_active", "loc_dual")}
    Ax = np.zeros(nc)
    for _ in range(M):
        F = rng.standard_normal((n, n))
        Q = F.T @ F + n * np.eye(n)
        G = rng.standard_normal((ne, n))
        H = np.vstack([np.eye(n), -np.eye(n)])
        A = rng.standard_normal((nc, n))
        zf = rng.standard_normal(n)

        data["Q"].append(Q)
        data["p"].append(rng.standard_normal(n))
        data["G"].append(G)
        data["g"].append(G @ zf)
        data["H"].append(H)
        data["h"].append(5 * np.ones(2 * n))
        data["A"].append(A)
        data["z0"].append(zf)
        data["loc_active"].append(np.zeros(2 * n, bool))
        data["loc_dual"].append(np.zeros(ne + 2 * n))
        Ax += A @ zf

    data["b"] = Ax + 1.0                       # z0 strictly feasible for the coupling
    data["coup_active"] = np.zeros(nc, bool)
    data["coup_dual"] = np.zeros(nc)
    return data


def _cli(argv=None):
    ap = argparse.ArgumentParser(
        prog="dmpc.py",
        description="Distributed active-set QP solver: closed-loop platoon demo.")
    ap.add_argument("steps", nargs="?", type=int, default=50,
                    help="closed-loop steps to simulate (default: 10)")
    ap.add_argument("-N", "--horizon", type=int, default=20,
                    help="prediction horizon (default: 20)")
    ap.add_argument("-M", "--agents", type=int, default=5,
                    help="number of followers (default: 5)")
    ap.add_argument("--ts", type=float, default=1.0,
                    help="sample time in seconds (default: 1.0)")
    ap.add_argument("--save", nargs="?", const="platoon.png", metavar="PATH",
                    help="write the figure to PATH instead of opening a window")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    ap.add_argument("--quiet", action="store_true",
                    help="summary only, no per-step lines")
    ap.add_argument("--skip-random", action="store_true",
                    help="skip the random-QP sanity check")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _cli()

    if not args.skip_random:
        data = _random_problem()
        sol = solve(data)
        print(f"random QP: {sol['status']} in {sol['iter']} iterations, "
              f"rho = {kkt_residual(data, sol):.3e}")

    out = run_platoon(n_iter=args.steps, N=args.horizon, M=args.agents,
                      Ts=args.ts, verbose=not args.quiet)

    print(f"\nworst rho over {args.steps} steps: {out['rho'].max():.3e}")
    print(f"iterations: mean {out['iter'].mean():.1f}, max {out['iter'].max()}")
    print(f"doubles/step: central {out['central'].mean():.0f}, "
          f"local {out['local'].mean():.0f}")

    if not args.no_plot:
        try:
            plot_run(out, path=args.save)
        except ImportError:
            print("matplotlib not installed; skipping plots")
