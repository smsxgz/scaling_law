# EVOLVE-BLOCK-START
import numpy as np
# Retain the powerful global optimization algorithm, which is essential for
# fitting complex, non-convex scaling laws.
from scipy.optimize import differential_evolution

def scaling_law_func(data_points, hyperparams):
    """
    Implements the generalized mean model for scaling laws, which is highly flexible
    for capturing different types of interactions between model size (P) and
    data size (D). This form has been shown to be highly effective.

    L(P, D) = E + ( (A*P**-alpha)**rho + (B*D**-beta)**rho )**(1/rho)

    This 6-parameter model can represent various physical interactions:
    - rho = 1:  Arithmetic mean (simple additive contributions).
    - rho -> 0: Geometric mean (multiplicative effects).
    - rho = -1: Harmonic mean (strong bottleneck behavior, where the smaller term
                contributes most to the loss).
    This adaptability is crucial for accurately modeling data-constrained scenarios.
    """
    data_points = np.atleast_2d(data_points)
    P = data_points[:, 0]
    D = data_points[:, 1]

    # Unpack the 6 hyperparameters for clarity
    E, A, alpha, B, beta, rho = hyperparams

    # Using np.errstate to handle potential overflows during optimization
    with np.errstate(over='ignore', invalid='ignore'):
        # Calculate the individual loss components from model and data size
        param_term = A * (P ** -alpha)
        data_term = B * (D ** -beta)

    # For numerical stability, ensure terms are positive before exponentiation.
    # This prevents complex numbers and issues with negative bases.
    stable_param_term = np.maximum(param_term, 1e-15)
    stable_data_term = np.maximum(data_term, 1e-15)

    # The generalized mean is numerically unstable when rho is near zero.
    # We handle this by explicitly implementing the limit case (geometric mean)
    # for improved stability.
    if abs(rho) < 1e-7:
        # This form is more stable than (t1^r + t2^r)^(1/r) for small rho.
        combined_term = np.exp(0.5 * (np.log(stable_param_term) + np.log(stable_data_term)))
    else:
        # Use np.errstate again for the final combination step.
        with np.errstate(over='ignore', invalid='ignore'):
            base = (stable_param_term ** rho) + (stable_data_term ** rho)
            # Clip the base to prevent issues with the outer exponent (e.g., root of negative).
            stable_base = np.maximum(base, 1e-15)
            combined_term = stable_base ** (1.0 / rho)

    predicted_loss = E + combined_term
    return predicted_loss


def fit_scaling_law(data_points, loss_values):
    """
    Fits the generalized mean scaling law using Differential Evolution. This
    version is evolved to use a Huber loss function instead of Mean Squared Error,
    making the fit more robust to potential outliers in the small (N=16) dataset.
    """

    def huber_loss(residuals, delta=0.1):
        """
        Calculates the Huber loss, which is quadratic for small residuals and
        linear for large ones, combining the benefits of MSE and MAE.
        """
        is_small_error = np.abs(residuals) <= delta
        squared_loss = 0.5 * (residuals ** 2)
        linear_loss = delta * (np.abs(residuals) - 0.5 * delta)
        return np.where(is_small_error, squared_loss, linear_loss)

    # The objective is to minimize the mean Huber loss.
    def objective(hyperparams):
        predicted_loss = scaling_law_func(data_points, hyperparams)

        # A critical guard: Penalize heavily if parameters lead to non-finite
        # predictions, guiding the optimizer towards stable regions.
        if not np.all(np.isfinite(predicted_loss)):
            return 1e12

        residuals = predicted_loss - loss_values
        # Using a delta of 0.1, which is a reasonable threshold for errors
        # in the typical loss range of LLMs.
        loss = np.mean(huber_loss(residuals, delta=0.1))
        return loss

    # The irreducible loss 'E' must be less than the minimum observed loss.
    min_loss = np.min(loss_values) if len(loss_values) > 0 else 1.0

    # Well-defined bounds are crucial for Differential Evolution's performance.
    # Parameter order: [E, A, alpha, B, beta, rho]
    bounds = [
        (0.0, min_loss * 0.999),  # E: Positive, below min observed loss.
        (1e-9, 1e12),            # A: Positive, wide range to accommodate scale.
        (1e-9, 2.0),             # alpha: Positive exponent, typical range.
        (1e-9, 1e12),            # B: Positive, wide range to accommodate scale.
        (1e-9, 2.0),             # beta: Positive exponent, typical range.
        (-10.0, 10.0)            # rho: Interaction term, wide range.
    ]

    # Differential Evolution finds a global minimum. `polish=True` refines
    # the solution with a local optimizer for precision.
    result = differential_evolution(
        objective,
        bounds,
        maxiter=1500,      # Increased iterations for the robust loss function.
        tol=1e-9,
        polish=True,
        seed=42            # For reproducibility.
    )

    return result.x
# EVOLVE-BLOCK-END