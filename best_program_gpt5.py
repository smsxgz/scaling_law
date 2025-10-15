# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, hyperparams):
    """
    Hybrid union-additive dual-saturation scaling law (up to 7 parameters):
      AP = 1/(1 + (P/p0)^alpha), AD = 1/(1 + (D/d0)^beta)
      S  = AP + AD - gamma * AP * AD
      L(P,D) = L_inf + Delta * S

    - data_points: (N,2+) array, first two cols are [params, tokens]
    - hyperparams: shape (6,) or (7,) or (T,6/7) with
        [L_inf, Delta, p0, d0, alpha, beta] (+ optional gamma in [0,1])
      If gamma absent, gamma=0 (pure additive).
    Returns: (N,) or (N,T) predicted loss
    """
    X = np.atleast_2d(np.asarray(data_points))
    P = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)[:, None]
    D = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)[:, None]

    par = np.asarray(hyperparams, dtype=float)
    if par.ndim == 1:
        par = par[None, :]
    # Pad with defaults to 7 (gamma default=0)
    if par.shape[1] < 7:
        defaults = np.array([1.2, 3.0, 1e9, 1e11, 0.5, 0.5, 0.0], dtype=float)
        pad = np.tile(defaults, (par.shape[0], 1))
        pad[:, :par.shape[1]] = par
        par = pad

    L_inf = par[:, 0][None, :]
    Delta = par[:, 1][None, :]
    p0    = np.clip(par[:, 2][None, :], 1.0, None)
    d0    = np.clip(par[:, 3][None, :], 1.0, None)
    alpha = np.clip(par[:, 4][None, :], 1e-8, None)
    beta  = np.clip(par[:, 5][None, :], 1e-8, None)
    gamma = np.clip(par[:, 6][None, :], 0.0, 1.0)

    eP = np.clip(alpha * (np.log(P) - np.log(p0)), -60.0, 60.0)
    eD = np.clip(beta  * (np.log(D) - np.log(d0)), -60.0, 60.0)
    uP = np.exp(eP)
    uD = np.exp(eD)
    AP = 1.0 / (1.0 + uP)
    AD = 1.0 / (1.0 + uD)
    S  = AP + AD - gamma * AP * AD
    pred = L_inf + Delta * S
    return pred[:, 0] if pred.shape[1] == 1 else pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit up to 7-parameter hybrid dual-saturation law using L-BFGS-B with
    analytic gradients, Huber objective, centered log-scale parameterization
    for p0,d0, and small priors. Returns:
      [L_inf, Delta, p0, d0, alpha, beta, gamma]
    """
    X = np.atleast_2d(np.asarray(data_points))
    y = np.asarray(loss_values, dtype=float)
    P = np.clip(X[:, 0].astype(float), 1.0, None)
    D = np.clip(X[:, 1].astype(float), 1.0, None)
    Y = y[:, None] if y.ndim == 1 else y
    T = Y.shape[1]

    # Stable transforms
    def softplus(z):  # positive
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    def sp_grad(z):   # d softplus / dz
        return 0.5 * (1.0 + np.tanh(0.5 * z))
    def sp_inv(x):    # inverse softplus
        x = np.maximum(x, 1e-12); return np.log(np.expm1(x))
    def sgm(z):       # sigmoid in [0,1]
        return 0.5 * (1.0 + np.tanh(0.5 * z))
    def sgm_grad(z):  # d sigmoid / dz
        s = sgm(z); return s * (1.0 - s)
    def logit(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6); return np.log(p / (1.0 - p))

    logP = np.log(P); logD = np.log(D)
    cP = float(np.mean(logP)); cD = float(np.mean(logD))  # centers for p0,d0

    def make_objective(yc):
        yc = yc.astype(float)
        huber_delta = 0.45
        lam_scale = 2e-4   # centered log-scale prior (u2,u3)
        lam_exp   = 1.5e-3 # exponent prior to ~0.5
        lam_ridge = 1e-8   # mild ridge on Delta
        lam_gamma = 8e-4   # keep gamma moderate
        ymin, ymax = float(np.min(yc)), float(np.max(yc))
        amp = max(1e-3, ymax - ymin)
        L_inf_target = max(1e-6, ymin - 0.1 * amp)
        lam_Linf = 2e-3

        def map(u):
            L_inf = softplus(u[0])
            Delta = softplus(u[1])
            p0 = np.exp(cP + u[2])
            d0 = np.exp(cD + u[3])
            alpha = softplus(u[4])
            beta  = softplus(u[5])
            gamma = sgm(u[6])
            return L_inf, Delta, p0, d0, alpha, beta, gamma

        def val_grad(u):
            L_inf, Delta, p0, d0, alpha, beta, gamma = map(u)
            lp0 = cP + u[2]; ld0 = cD + u[3]

            eP = np.clip(alpha * (logP - lp0), -60.0, 60.0)
            eD = np.clip(beta  * (logD - ld0), -60.0, 60.0)
            uP = np.exp(eP); uD = np.exp(eD)
            AP = 1.0 / (1.0 + uP); AD = 1.0 / (1.0 + uD)
            S  = AP + AD - gamma * AP * AD
            pred = L_inf + Delta * S
            err = pred - yc

            ae = np.abs(err); m = (ae <= huber_delta)
            loss = np.where(m, 0.5 * err * err, huber_delta * (ae - 0.5 * huber_delta))
            dLdp = np.where(m, err, huber_delta * np.sign(err))
            base = float(np.mean(loss))

            dL_Linf  = np.mean(dLdp)
            dL_Delta = np.mean(dLdp * S)

            # dS/dAP = 1 - gamma*AD; dS/dAD = 1 - gamma*AP; dS/dgamma = -AP*AD
            dS_dAP = (1.0 - gamma * AD)
            dS_dAD = (1.0 - gamma * AP)
            dS_dg  = -(AP * AD)

            # dAP/duP = -AP^2; duP/dalpha = uP*(logP - lp0); duP/dlp0 = -alpha*uP
            # dAD/duD = -AD^2; duD/dbeta  = uD*(logD - ld0); duD/dld0 = -beta*uD
            dL_alpha = np.mean(dLdp * (Delta * dS_dAP * (-AP * AP) * (uP * (logP - lp0))))
            dL_lp0   = np.mean(dLdp * (Delta * dS_dAP * (-AP * AP) * (uP * (-alpha))))
            dL_beta  = np.mean(dLdp * (Delta * dS_dAD * (-AD * AD) * (uD * (logD - ld0))))
            dL_ld0   = np.mean(dLdp * (Delta * dS_dAD * (-AD * AD) * (uD * (-beta))))
            dL_gamma = np.mean(dLdp * (Delta * dS_dg))

            # Regularization
            reg = 0.0
            reg += lam_scale * (u[2]**2 + u[3]**2)
            reg += lam_exp * ((alpha - 0.5)**2 + (beta - 0.5)**2)
            reg += lam_ridge * (Delta**2)
            reg += lam_Linf * (L_inf - L_inf_target)**2
            reg += lam_gamma * (gamma - 0.4)**2

            # Gradients wrt raw u
            g = np.zeros_like(u)
            g[0] = dL_Linf * sp_grad(u[0]) + 2.0 * lam_Linf * (L_inf - L_inf_target) * sp_grad(u[0])
            g[1] = dL_Delta * sp_grad(u[1]) + 2.0 * lam_ridge * Delta * sp_grad(u[1])
            g[2] = dL_lp0 + 2.0 * lam_scale * u[2]
            g[3] = dL_ld0 + 2.0 * lam_scale * u[3]
            g[4] = dL_alpha * sp_grad(u[4]) + 2.0 * lam_exp * (alpha - 0.5) * sp_grad(u[4])
            g[5] = dL_beta  * sp_grad(u[5])  + 2.0 * lam_exp * (beta  - 0.5) * sp_grad(u[5])
            g[6] = dL_gamma * sgm_grad(u[6]) + 2.0 * lam_gamma * (gamma - 0.4) * sgm_grad(u[6])
            return base + reg, g
        return val_grad

    params_out = np.zeros((T, 7), dtype=float)
    for t in range(T):
        yc = Y[:, t]
        ymin, ymax = float(np.min(yc)), float(np.max(yc))
        amp = max(1e-3, ymax - ymin)
        L0 = max(1e-6, ymin - 0.1 * amp)
        Dlt0 = max(1e-3, 0.6 * amp)

        alpha_opts = [0.4, 0.7]
        beta_opts  = [0.4, 0.7]
        gamma_opts = [0.0, 0.5]
        p_scales   = [0.8, 1.2]
        d_scales   = [0.8, 1.2]

        starts = []
        for a in alpha_opts:
            for b in beta_opts:
                for g in gamma_opts:
                    for ps in p_scales:
                        for ds in d_scales:
                            lp0 = cP + np.log(ps)
                            ld0 = cD + np.log(ds)
                            AP = 1.0 / (1.0 + np.exp(np.clip(a * (logP - lp0), -60.0, 60.0)))
                            AD = 1.0 / (1.0 + np.exp(np.clip(b * (logD - ld0), -60.0, 60.0)))
                            Z = AP + AD - g * AP * AD
                            A = np.vstack([np.ones_like(Z), Z]).T
                            try:
                                coef, _, _, _ = np.linalg.lstsq(A, yc, rcond=None)
                                L_init = float(max(1e-6, coef[0]))
                                Dlt_init = float(max(1e-6, coef[1]))
                            except Exception:
                                L_init, Dlt_init = L0, Dlt0
                            starts.append([L_init, Dlt_init, np.log(ps), np.log(ds), a, b, g])
                            if len(starts) >= 6: break
                        if len(starts) >= 6: break
                    if len(starts) >= 6: break
                if len(starts) >= 6: break
            if len(starts) >= 6: break

        raw_starts = [np.array([sp_inv(s[0]), sp_inv(s[1]),
                                s[2], s[3],
                                sp_inv(s[4]), sp_inv(s[5]),
                                logit(s[6])], dtype=float) for s in starts]

        obj = make_objective(yc)
        best_val = np.inf
        best_params = np.array([L0, Dlt0, np.exp(cP), np.exp(cD), 0.5, 0.5, 0.0], dtype=float)

        for u0 in raw_starts:
            res = minimize(lambda u: obj(u)[0], u0, method='L-BFGS-B',
                           jac=lambda u: obj(u)[1],
                           options={'maxiter': 600, 'ftol': 1e-9})
            if res.success:
                u = res.x
                L_inf = softplus(u[0]); Delta = softplus(u[1])
                p0 = np.exp(cP + u[2]); d0 = np.exp(cD + u[3])
                alpha = softplus(u[4]); beta = softplus(u[5]); gamma = sgm(u[6])
                cand = np.array([L_inf, Delta, p0, d0, alpha, beta, gamma], dtype=float)
                val = res.fun
            else:
                val = obj(u0)[0]
                cand = np.array([softplus(u0[0]), softplus(u0[1]),
                                 np.exp(cP + u0[2]), np.exp(cD + u0[3]),
                                 softplus(u0[4]), softplus(u0[5]), sgm(u0[6])], dtype=float)
            if val < best_val:
                best_val = val; best_params = cand

        params_out[t, :] = best_params

    return params_out[0] if T == 1 else params_out
# EVOLVE-BLOCK-END