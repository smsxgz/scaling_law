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
    which corresponds to the fixed-effect part of the linear mixed model 
    fitted in the fit_scaling_law function.

    Args:
        data_points (np.ndarray): (N, 3) array with columns [problem_id, params, tokens]
        hyperparams (np.ndarray): Array of 3 fixed-effect parameters: [C, alpha, beta]

    Returns:
        np.ndarray: (N,) array of predicted (fixed-effect only) loss values
    """
    # Unpack params and tokens from data_points
    # problem_id is not used in this function as it only calculates fixed effects
    params = data_points[:, 1]
    tokens = data_points[:, 2]
    
    # Unpack hyperparameters
    C, alpha, beta = hyperparams
    
    # Calculate predicted loss
    # L = C * N^(-alpha) * D^(-beta)
    # We assume params and tokens are always positive based on data description
    predicted_loss = C * (params ** -alpha) * (tokens ** -beta)
    
    return predicted_loss

def fit_scaling_law(data_points, loss_values):
    """
    Fits a Linear Mixed-Effects Model (LMM) to find the optimal *fixed-effect* hyperparameters.

    This model operates in log-space to linearize the power law:
    log(L) = log(C) - alpha*log(N) - beta*log(D) + (random_intercept_for_problem_id)
    
    The model it fits is:
    log_loss ~ log_params + log_tokens + (1 | problem_id)

    Args:
        data_points (np.ndarray): (N, 3) array with columns [problem_id, params, tokens]
        loss_values (np.ndarray): (N,) array of corresponding true loss values

    Returns:
        np.ndarray: Optimized *fixed-effect* hyperparameters [C, alpha, beta]
    """
    
    # 1. Unpack and transform data to log-space
    try:
        problem_id = data_points[:, 0]
        # Use np.log for numerical stability with large values
        log_params = np.log(data_points[:, 1])
        log_tokens = np.log(data_points[:, 2])
        log_loss = np.log(loss_values)
    except Exception as e:
        print(f"Error during log transformation: {e}. Check for non-positive values.")
        return np.array([1.0, 0.0, 0.0]) # Returning dummy/invalid parameters

    # 2. Assemble data into a pandas DataFrame, as required by the statsmodels formula API
    # Convert problem_id to 'category' or 'str' type so statsmodels 
    # treats it as a grouping variable, not a continuous number.
    df = pd.DataFrame({
        'log_loss': log_loss,
        'log_params': log_params,
        'log_tokens': log_tokens,
        'problem_id': pd.Categorical(problem_id)
    })

    # 3. Define and fit the Linear Mixed-Effects Model
    # The formula "log_loss ~ log_params + log_tokens" defines the fixed effects.
    # The 'groups' argument defines the random effect 
    # (we want a random intercept for each 'problem_id').
    try:
        mixed_lm_model = smf.mixedlm(formula="log_loss ~ log_params + log_tokens", 
                                     data=df, 
                                     groups=df["problem_id"])
        
        # Fit using the 'lbfgs' optimizer, which is often robust.
        fit_result = mixed_lm_model.fit(method=["lbfgs"])
        
    except Exception as e:
        print(f"Failed to fit Linear Mixed-Effects Model: {e}")
        # If LMM fails, a simple OLS (Ordinary Least Squares) could be 
        # returned as a fallback, but here we just return a dummy array.
        return np.array([1.0, 0.0, 0.0])

    # 4. Extract the fixed-effect parameters from the fit result
    # fit_result.params is a Series containing keys 'Intercept', 'log_params', 'log_tokens'
    
    # Our model is: 
    # log(L) = log(C) - alpha*log(N) - beta*log(D)
    
    # statsmodels fits:
    # log_loss = Intercept + (coef_log_params)*log_params + (coef_log_tokens)*log_tokens
    
    # Therefore, we can establish the following relationships:
    # Intercept = log(C)       => C = exp(Intercept)
    # coef_log_params = -alpha => alpha = -coef_log_params
    # coef_log_tokens = -beta  => beta = -coef_log_tokens
    
    fixed_effects_params = fit_result.params
    
    global_intercept = fixed_effects_params['Intercept']
    coef_log_params = fixed_effects_params['log_params']
    coef_log_tokens = fixed_effects_params['log_tokens']
    
    # Convert back to our scaling_law_func's [C, alpha, beta] format
    C = np.exp(global_intercept)
    alpha = -coef_log_params
    beta = -coef_log_tokens
    
    # Return only the fixed-effect hyperparameters
    return np.array([C, alpha, beta])

# EVOLVE-BLOCK-END