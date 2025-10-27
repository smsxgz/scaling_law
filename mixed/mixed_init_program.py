# EVOLVE-BLOCK-START

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# --- DECLARE THE CALIBRATION MODE ---
# This tells the evaluator how to "quickly estimate" the random effect.
# 'multiplicative': L_total = L_fixed * K_i  (Use for log-linear models)
# 'additive':       L_total = L_fixed + C_i  (Use for simple additive models)
CALIBRATION_MODE = 'multiplicative'
# ----------------------------------------

def scaling_law_func(data_points, hyperparams):
    """
    Calculates the predicted loss from the fixed-effect (global) part of the scaling law.
    
    This function implements the power law: L = C * (params**-alpha) * (tokens**-beta)
    
    Args:
        data_points (np.ndarray): (N, 3) array with columns [problem_id, params, tokens]
        hyperparams (np.ndarray): Array of 3 fixed-effect parameters: [C, alpha, beta]

    Returns:
        np.ndarray: (N,) array of predicted (fixed-effect only) loss values
    """
    # Explicitly cast object-array slices to float64 before math.
    params = data_points[:, 1].astype(float) 
    tokens = data_points[:, 2].astype(float)
    
    C, alpha, beta = hyperparams
    
    # This will now correctly return a float64 array
    predicted_loss = C * (params ** -alpha) * (tokens ** -beta)
    
    return predicted_loss

def fit_scaling_law(data_points, loss_values):
    """
    Fits a Linear Mixed-Effects Model (LMM) in log-space.
    
    log(L) = log(C) - alpha*log(P) - beta*log(D) + (random_intercept_for_problem_id)
    
    Args:
        data_points (np.ndarray): (N, 3) array with columns [problem_id, params, tokens]
        loss_values (np.ndarray): (N,) array of corresponding true loss values

    Returns:
        np.ndarray: Optimized *fixed-effect* hyperparameters [C, alpha, beta]
    """
    
    try:
        problem_id = data_points[:, 0]
        
        # Explicitly cast object-array slices and loss_values to float64
        # *before* passing them to np.log.
        log_params = np.log(data_points[:, 1].astype(float))
        log_tokens = np.log(data_points[:, 2].astype(float))
        log_loss = np.log(loss_values.astype(float))
        
    except Exception as e:
        # This print statement is what you saw in your error log
        print(f"Error during log transformation: {e}. Check for non-positive values.")
        return np.array([1.0, 0.0, 0.0])

    df = pd.DataFrame({
        'log_loss': log_loss,
        'log_params': log_params,
        'log_tokens': log_tokens,
        'problem_id': pd.Categorical(problem_id)
    })

    try:
        mixed_lm_model = smf.mixedlm(formula="log_loss ~ log_params + log_tokens", 
                                     data=df, 
                                     groups=df["problem_id"])
        
        fit_result = mixed_lm_model.fit(method=["lbfgs"])
        
    except Exception as e:
        print(f"Failed to fit Linear Mixed-Effects Model: {e}")
        return np.array([1.0, 0.0, 0.0])

    fixed_effects_params = fit_result.params
    
    global_intercept = fixed_effects_params['Intercept']
    coef_log_params = fixed_effects_params['log_params']
    coef_log_tokens = fixed_effects_params['log_tokens']
    
    C = np.exp(global_intercept)
    alpha = -coef_log_params
    beta = -coef_log_tokens
    
    return np.array([C, alpha, beta])

# EVOLVE-BLOCK-END