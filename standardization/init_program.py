# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios.

Model Form:
logprob_zscore = (coeff_P * log10(params)) + (coeff_T * log10(tokens)) + intercept
"""
import numpy as np


def scaling_law_func(data_points, hyperparams):
    """
    Applies the fitted linear model to log-transformed data.
    
    Args:
        data_points: (N, 2) array of [params, tokens]
        hyperparams: (3,) array of [coeff_logP, coeff_logT, intercept]
        
    Returns:
        Predicted logprob_zscore values
    """
    X_raw = np.atleast_2d(np.asarray(data_points))
    
    # 1. Transform features: [P, T] -> [log10(P), log10(T)]
    # Use 1e-9 to prevent log(0) errors (though params/tokens should always be > 0)
    X_log = np.log10(np.maximum(X_raw, 1e-9))
    
    # 2. Add bias (intercept) term
    # np.hstack adds a column of 1s alongside [logP, logT]
    X_with_bias = np.hstack([X_log, np.ones((X_log.shape[0], 1))])
    
    # 3. Apply the linear model
    # hyperparams is a (3,) array: [coeff_logP, coeff_logT, intercept]
    params_vec = np.asarray(hyperparams)
    
    # (N, 3) @ (3,) -> (N,)
    pred = np.dot(X_with_bias, params_vec)
    
    return pred


def fit_scaling_law(data_points, logprob_zscore_values):
    """
    Fits a linear model using exact Linear Least Squares (np.linalg.lstsq).
    This is extremely fast and does not require iteration.
    
    Args:
        data_points: (N, 2) array of [params, tokens]
        logprob_zscore_values: (N,) array of target values
        
    Returns:
        (3,) array of optimized hyperparameters: [coeff_logP, coeff_logT, intercept]
    """
    X_raw = np.atleast_2d(np.asarray(data_points))
    y = np.asarray(logprob_zscore_values)
    
    N, F = X_raw.shape # F should be 2

    # 1. Transform features: [P, T] -> [log10(P), log10(T)]
    X_log = np.log10(np.maximum(X_raw, 1e-9))

    # 2. Create the design matrix (X'), adding the bias term
    X_with_bias = np.hstack([X_log, np.ones((N, 1))])
    
    # 3. Solve using Linear Least Squares
    # This solves the equation X_with_bias * b = y, solving for b
    # b will be an (F+1,) array, i.e., (3,)
    try:
        # rcond=None uses numpy's default for future compatibility
        params_opt, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback in case of a singular matrix or other linear algebra error
        print("Warning: Linear regression failed due to LinAlgError. Returning zeros.")
        params_opt = np.zeros(F + 1)
    
    # params_opt is now [coeff_logP, coeff_logT, intercept]
    return params_opt
# EVOLVE-BLOCK-END