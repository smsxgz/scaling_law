# --- DECLARE THE CALIBRATION MODE ---
# This tells the evaluator how to "quickly estimate" the random effect.
# 'multiplicative': L_total = L_fixed * K_i
# 'additive':       L_total = L_fixed + C_i
CALIBRATION_MODE = 'additive'
# ----------------------------------------

# EVOLVE-BLOCK-START
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

def scaling_law_func(data_points, hyperparams):
    """
    Calculates the predicted loss from the fixed-effect (global) part of the scaling law.
    
    This function implements the simple log-linear additive law:
    L_fixed = A * log(params) + B * log(tokens)
    
    This corresponds to the model:
    loss_ij = (A * log(params_i) + B * log(tokens_i)) + u_j
    
    Args:
        data_points (np.ndarray): (N, 3) array with columns [problem_id, params, tokens]
        hyperparams (np.ndarray): Array of 2 fixed-effect parameters: [A, B]

    Returns:
        np.ndarray: (N,) array of predicted (fixed-effect only) loss values
    """
    params = data_points[:, 1].astype(float) 
    tokens = data_points[:, 2].astype(float)
    
    A, B = hyperparams
    
    # --- Stability Guards ---
    # Clip inputs to avoid log(0) -> -inf
    safe_params = np.clip(params, 1e-9, None)
    safe_tokens = np.clip(tokens, 1e-9, None)
    # --- End Guards ---
    
    log_params = np.log(safe_params)
    log_tokens = np.log(safe_tokens)
    
    # Return the simple linear combination (no L_inf intercept)
    return A * log_params + B * log_tokens

def fit_scaling_law(data_points, loss_values):
    """
    Fits an Additive Linear Mixed-Effects Model (LMM) using log features.
    
    Model: loss_ij = A*log(P_i) + B*log(D_i) + u_j
    
    This function fits the model by finding the coefficients A and B, while
    modeling u_j as a random intercept for each 'problem_id'.
    
    Crucially, it does *not* fit a global intercept ('-1' in formula),
    forcing u_j to absorb the entire per-problem baseline, as requested.
    
    Args:
        data_points (np.ndarray): (N, 3) array with columns [problem_id, params, tokens]
        loss_values (np.ndarray): (N,) array of corresponding true loss values

    Returns:
        np.ndarray: Optimized *fixed-effect* hyperparameters [A, B]
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Suppress statsmodels warnings
        
        try:
            problem_id = data_points[:, 0]
            
            # Create the fixed-effect regressors (the X matrix)
            # Clip to avoid log(0)
            safe_params = np.clip(data_points[:, 1].astype(float), 1e-9, None)
            safe_tokens = np.clip(data_points[:, 2].astype(float), 1e-9, None)

            log_params = np.log(safe_params)
            log_tokens = np.log(safe_tokens)
            
            df = pd.DataFrame({
                'loss': loss_values.astype(float),
                'log_params': log_params,
                'log_tokens': log_tokens,
                'problem_id': pd.Categorical(problem_id)
            })

            # Handle potential inf/-inf if params/tokens were 0 (though clipped)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=['loss', 'log_params', 'log_tokens', 'problem_id'], inplace=True)
            
            if df.empty:
                print("Warning: DataFrame is empty after cleaning inf/nan.")
                return np.array([0.0, 0.0])

            # Formula: loss ~ log_params + log_tokens - 1
            # We use "- 1" to EXPLICITLY remove the global intercept.
            # This forces the random intercepts (u_j) to capture the
            # entire per-problem baseline, precisely matching the prompt's requirement.
            formula = "loss ~ log_params + log_tokens - 1"
            
            mixed_lm_model = smf.mixedlm(formula=formula,
                                         data=df, 
                                         groups=df["problem_id"])
            
            # Fit the model
            fit_result = mixed_lm_model.fit(method=["lbfgs"])
            
            fixed_effects_params = fit_result.params
            
            # The parameters are directly the coefficients A and B
            A = fixed_effects_params.get('log_params', 0.0)
            B = fixed_effects_params.get('log_tokens', 0.0)
            
            return np.array([A, B])

        except Exception as e:
            print(f"Failed to fit Linear Mixed-Effects Model: {e}")
            # Return neutral parameters for [A, B]
            return np.array([0.0, 0.0])

# EVOLVE-BLOCK-END