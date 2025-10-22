# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize

# The model is L_total = L_fixed + C_i, so the random effect is additive.
CALIBRATION_MODE = 'additive'

def scaling_law_func(data_points, hyperparams):
    """
    Fixed-effect prediction using a generalized mean of two power laws.
    This form is flexible and can model additive, multiplicative-like, and max-like behaviors.
    Model: L(P,D) = L_inf + [ (A * P^-alpha)^r + (B * D^-beta)^r ]^(1/r)
    """
    # This function is identical to the top-performing program as it represents a
    # state-of-the-art, numerically stable functional form.
    P = data_points[:, 1].astype(np.float64)
    D = data_points[:, 2].astype(np.float64)
    A, alpha, B, beta, L_inf, r = hyperparams

    # Clamp inputs and parameters to ensure numerical stability.
    P = np.maximum(P, 1e-12)
    D = np.maximum(D, 1e-12)
    A = max(A, 1e-18)
    B = max(B, 1e-18)
    r = max(r, 1e-6)

    # Use log-space calculations for stability (log-sum-exp trick).
    logP = np.log(P)
    logD = np.log(D)
    
    term1_log = np.log(A) - alpha * logP
    term2_log = np.log(B) - beta * logD

    r_term1_log = r * term1_log
    r_term2_log = r * term2_log
    
    max_val = np.maximum(r_term1_log, r_term2_log)
    log_sum_exp = max_val + np.log(np.exp(r_term1_log - max_val) + np.exp(r_term2_log - max_val))
    
    power_law_component = np.exp(log_sum_exp / r)
    
    return L_inf + power_law_component

def fit_scaling_law(data_points, loss_values):
    """
    Fits fixed effects using a multi-start optimization strategy to improve robustness
    against local minima in the complex, non-convex loss landscape. It runs the
    optimization from several plausible starting points and returns the best result.
    The core objective function still profiles out the per-problem random intercepts.
    """
    problem_ids = data_points[:, 0]
    params = data_points[:, 1].astype(np.float64)
    tokens = data_points[:, 2].astype(np.float64)
    y = loss_values.astype(np.float64)

    # Pre-process group information for efficient residual calculation.
    unique_ids, group_indices = np.unique(problem_ids, return_inverse=True)
    num_groups = len(unique_ids)
    group_counts = np.bincount(group_indices, minlength=num_groups).astype(np.float64)
    min_y = np.min(y)

    # --- Objective Function with Profiled Random Effects ---
    # This is the core SSE objective, proven effective in top models.
    def objective(theta):
        try:
            pred = scaling_law_func(data_points, theta)
            residuals = y - pred
            
            sum_by_group = np.bincount(group_indices, weights=residuals, minlength=num_groups)
            mean_by_group = np.divide(sum_by_group, group_counts, out=np.zeros_like(sum_by_group), where=group_counts!=0)
            
            adjusted_residuals = residuals - mean_by_group[group_indices]
            
            sse = np.dot(adjusted_residuals, adjusted_residuals)
            return sse if np.isfinite(sse) else 1e20
        except (ValueError, OverflowError):
            return 1e20

    # --- Helper to generate smart initial guesses ---
    def make_guess(L_inf_guess):
        remainder = y - L_inf_guess
        positive_mask = remainder > 1e-9
        A_guess, B_guess, alpha_guess, beta_guess = 5.0, 5.0, 0.2, 0.2 # More stable fallbacks
        if np.sum(positive_mask) > 3:
            try:
                log_params_pos = np.log(np.maximum(params[positive_mask], 1e-12))
                log_tokens_pos = np.log(np.maximum(tokens[positive_mask], 1e-12))
                Y_log = np.log(remainder[positive_mask])
                X_log = np.column_stack([np.ones(np.sum(positive_mask)), log_params_pos, log_tokens_pos])
                coeffs, _, _, _ = np.linalg.lstsq(X_log, Y_log, rcond=None)
                alpha_guess = float(np.clip(-coeffs[1], 1e-4, 2.0))
                beta_guess = float(np.clip(-coeffs[2], 1e-4, 2.0))

                f1 = params[positive_mask]**(-alpha_guess)
                f2 = tokens[positive_mask]**(-beta_guess)
                X_lin = np.column_stack([f1, f2])
                coeffs_lin, _, _, _ = np.linalg.lstsq(X_lin, remainder[positive_mask], rcond=None)
                A_guess = float(max(coeffs_lin[0], 1e-9))
                B_guess = float(max(coeffs_lin[1], 1e-9))
            except (np.linalg.LinAlgError, ValueError):
                pass
        return np.array([A_guess, alpha_guess, B_guess, beta_guess, L_inf_guess, 1.0])

    # --- Multi-Start Strategy ---
    # Start 1: Primary guess based on robust quantile for L_inf
    L_inf_guess_1 = np.clip(np.quantile(y, 0.01), 0.0, min_y * 0.99 if min_y > 0 else 0)
    guess1 = make_guess(L_inf_guess_1)
    
    # Start 2: Alternative guess assuming L_inf is zero
    guess2 = make_guess(0.0)
    
    # Start 3: Primary guess but starting with a max-like behavior (high 'r')
    guess3 = np.copy(guess1)
    guess3[-1] = 4.5 # Start r near the upper bound

    initial_guesses = [guess1, guess2, guess3]
    
    best_result = None
    best_sse = np.inf
    
    bounds = [
        (1e-12, None),   # A > 0
        (1e-4, 2.0),     # 0 < alpha < 2
        (1e-12, None),   # B > 0
        (1e-4, 2.0),     # 0 < beta < 2
        (0.0, min_y),    # 0 <= L_inf < min_loss
        (1e-3, 5.0)      # 0 < r < 5
    ]
    
    # Run optimization from each starting point
    for initial_theta in initial_guesses:
        # Ensure initial guess is within bounds before optimization
        for i in range(len(initial_theta)):
            low, high = bounds[i]
            if low is not None: initial_theta[i] = max(initial_theta[i], low)
            if high is not None: initial_theta[i] = min(initial_theta[i], high)

        res = minimize(objective, initial_theta, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-9})
        
        if res.success and res.fun < best_sse:
            best_sse = res.fun
            best_result = res

    # Return the parameters from the best run, or the primary initial guess as a fallback.
    if best_result is not None and np.all(np.isfinite(best_result.x)):
        return best_result.x
    else:
        return guess1

# EVOLVE-BLOCK-END