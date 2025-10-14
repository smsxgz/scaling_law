#!/usr/bin/env python3

import os
import json
import logging
import random
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Definitions (Should match your evaluation script) ---
# It's crucial that params and tokens data is accurate.
# NOTE: I've corrected some of the pythia parameter counts for better accuracy.
MODELS_METADATA = {
    # Pythia
    "EleutherAI/pythia-14m": {"params": 14e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-31m": {"params": 31e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-70m": {"params": 70e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-160m": {"params": 160e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-410m": {"params": 410e6, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-1b": {"params": 1e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-1.4b": {"params": 1.4e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-2.8b": {"params": 2.8e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-6.9b": {"params": 6.9e9, "tokens": 300e9, "family": "pythia"},
    "EleutherAI/pythia-12b": {"params": 12e9, "tokens": 300e9, "family": "pythia"},
    # OPT
    "facebook/opt-125m": {"params": 125e6, "tokens": 180e9, "family": "opt"},
    "facebook/opt-350m": {"params": 350e6, "tokens": 180e9, "family": "opt"},
    "facebook/opt-1.3b": {"params": 1.3e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-2.7b": {"params": 2.7e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-6.7b": {"params": 6.7e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-13b": {"params": 13e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-30b": {"params": 30e9, "tokens": 180e9, "family": "opt"},
    "facebook/opt-66b": {"params": 66e9, "tokens": 180e9, "family": "opt"},
}


def load_logprob_data(base_dir: Path) -> pd.DataFrame:
    """
    Loads all _logprobs.json files recursively from all subdirectories
    and aggregates them into a single DataFrame.
    """
    all_data = []
    # --- FIX 1: Use rglob to search ALL subdirectories ---
    log_files = list(base_dir.rglob("*_logprobs.json"))
    
    if not log_files:
        raise FileNotFoundError(f"No '*_logprobs.json' files found anywhere inside {base_dir}. "
                              "Please run the evaluation script first.")
    
    logger.info(f"Found {len(log_files)} logprob files to process.")

    for log_file in log_files:
        # Reconstruct model name from filename, e.g., 'facebook_opt-1.3b' -> 'facebook/opt-1.3b'
        model_name_slug = log_file.name.replace("_logprobs.json", "")
        parts = model_name_slug.split('_')
        if len(parts) > 1 and parts[0] in ['facebook', 'EleutherAI']:
             model_name = f"{parts[0]}/{'_'.join(parts[1:])}"
        else:
             model_name = model_name_slug.replace('_', '/') # Fallback

        if model_name not in MODELS_METADATA:
            logger.warning(f"Skipping file {log_file.name}, as model '{model_name}' is not in METADATA.")
            continue
            
        with open(log_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        for idx, sample in enumerate(samples):
            # Create a robust unique ID for each question
            question_id = f"{sample.get('task', 'mmlu')}_{sample.get('doc_id', idx)}"
            
            # The logprobs for all choices
            logprobs = sample.get('logprobs')
            
            # --- FIX 2: Check for 'gold' first, then fall back to 'correct_answer' ---
            correct_answer_idx = sample.get('gold', sample.get('correct_answer'))
            
            # Ensure data is valid before processing
            if logprobs is None or correct_answer_idx is None:
                continue
            
            # MMLU tasks give integer index for the correct choice
            try:
                correct_answer_idx = int(correct_answer_idx)
                if correct_answer_idx >= len(logprobs):
                    continue
            except (ValueError, TypeError):
                continue # Skip if the index is not a valid integer

            # logloss is the negative log-probability of the correct answer
            correct_logprob = logprobs[correct_answer_idx]
            logloss = -float(correct_logprob)
            
            all_data.append({
                "question_id": question_id,
                "model_name": model_name,
                "logloss": logloss,
                "params": MODELS_METADATA[model_name]["params"],
                "tokens": MODELS_METADATA[model_name]["tokens"],
                "family": MODELS_METADATA[model_name]["family"],
            })
            
    return pd.DataFrame(all_data)


# --- Scaling Law Model Functions (Unchanged) ---

def log_linear_law(X, alpha, beta, gamma):
    """A. Linear-Log Law: -log(p) ~ α + β*log(N) + γ*log(T)"""
    log_N, log_T = X
    return alpha + beta * log_N + gamma * log_T

def chinchilla_law(X, L_inf, A, B):
    """B. Chinchilla-style Law: -log(p) ~ L_inf + A*N^(-alpha) + B*T^(-beta)"""
    alpha_fixed = 0.35
    beta_fixed = 0.35
    N, T = X
    return L_inf + A * (N ** -alpha_fixed) + B * (T ** -beta_fixed)


def analyze_single_question(df_question: pd.DataFrame, question_id: str):
    """Fits both scaling laws to the data for a single question and plots the results."""
    
    if len(df_question) < 5:
        logger.warning(f"Skipping question '{question_id}': only {len(df_question)} data points available.")
        return

    df_question = df_question.dropna().copy()
    y = df_question['logloss'].values
    N = df_question['params'].values
    T = df_question['tokens'].values
    log_N = np.log(N)
    log_T = np.log(T)

    # --- 1. Fit Law A: Linear-Log Model ---
    X_A = np.vstack([log_N, log_T]).T
    X_A = sm.add_constant(X_A, prepend=True)
    model_A = sm.OLS(y, X_A).fit()
    alpha, beta_N, gamma_T = model_A.params
    logger.info(f"\n--- Question: {question_id} ---")
    logger.info(f"[Law A] Fit: -log(p) ≈ {alpha:.2f} {beta_N:+.2f}*log(N) {gamma_T:+.2f}*log(T) (R²={model_A.rsquared:.3f})")

    # --- 2. Fit Law B: Chinchilla-style Model ---
    try:
        X_B = (N, T)
        initial_guesses = [min(y) * 0.9, 1.0, 1.0] 
        params_B, _ = curve_fit(chinchilla_law, X_B, y, p0=initial_guesses, maxfev=5000)
        L_inf, A, B = params_B
        logger.info(f"[Law B] Fit: -log(p) ≈ {L_inf:.2f} + {A:.2f}*N^(-0.35) + {B:.2e}*T^(-0.35)")
    except RuntimeError:
        logger.error(f"[Law B] Could not be fitted for question {question_id}.")
        params_B = None

    # --- 3. Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Scaling Law Analysis for MMLU Question: {question_id}', fontsize=16)

    # Plot vs. Model Parameters (N)
    ax1 = axes[0]
    T_mean_for_plot = np.exp(np.mean(log_T))
    N_smooth = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)
    
    pred_A_vs_N = alpha + beta_N * np.log(N_smooth) + gamma_T * np.log(T_mean_for_plot)
    ax1.plot(N_smooth, pred_A_vs_N, label='Fit A: Linear-Log', color='blue', linestyle='--')
    
    if params_B is not None:
        pred_B_vs_N = chinchilla_law((N_smooth, T_mean_for_plot), *params_B)
        ax1.plot(N_smooth, pred_B_vs_N, label='Fit B: Chinchilla', color='red', linestyle=':')

    for family, group in df_question.groupby('family'):
        ax1.scatter(group['params'], group['logloss'], label=family, alpha=0.8)

    ax1.set_xscale('log')
    ax1.set_xlabel('Model Parameters (N)')
    ax1.set_ylabel('LogLoss (-log p_correct)')
    ax1.set_title('LogLoss vs. Model Scale')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # Plot vs. Training Tokens (T)
    ax2 = axes[1]
    N_mean_for_plot = np.exp(np.mean(log_N))
    T_smooth = np.logspace(np.log10(T.min()), np.log10(T.max()), 100)
    
    pred_A_vs_T = alpha + beta_N * np.log(N_mean_for_plot) + gamma_T * np.log(T_smooth)
    ax2.plot(T_smooth, pred_A_vs_T, label='Fit A: Linear-Log', color='blue', linestyle='--')
    
    if params_B is not None:
        pred_B_vs_T = chinchilla_law((N_mean_for_plot, T_smooth), *params_B)
        ax2.plot(T_smooth, pred_B_vs_T, label='Fit B: Chinchilla', color='red', linestyle=':')

    for family, group in df_question.groupby('family'):
        ax2.scatter(group['tokens'], group['logloss'], label=family, alpha=0.8)

    ax2.set_xscale('log')
    ax2.set_xlabel('Training Tokens (T)')
    ax2.set_ylabel('LogLoss (-log p_correct)')
    ax2.set_title('LogLoss vs. Training Tokens')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    base_dir = Path("./eval_results")
    if not base_dir.exists():
        logger.error(f"Directory '{base_dir}' not found. Please run the evaluation script first.")
        return
        
    try:
        df = load_logprob_data(base_dir)
    except FileNotFoundError as e:
        logger.error(e)
        return

    if df.empty:
        logger.error("No valid logprob data could be loaded. "
                     "Check that JSON files exist and contain 'logprobs' and 'gold'/'correct_answer' keys.")
        return

    question_counts = df['question_id'].value_counts()
    min_models_threshold = int(df['model_name'].nunique() * 0.8)
    eligible_questions = question_counts[question_counts >= min_models_threshold].index.tolist()

    if not eligible_questions:
        logger.error(f"No questions were evaluated by at least {min_models_threshold} models. "
                     "Consider running more models or lowering the threshold.")
        return

    logger.info(f"Found {len(eligible_questions)} questions suitable for analysis.")
    
    num_to_analyze = min(5, len(eligible_questions)) # Increased to 5 examples
    selected_ids = random.sample(eligible_questions, num_to_analyze)

    for q_id in selected_ids:
        df_q = df[df['question_id'] == q_id]
        analyze_single_question(df_q, q_id)


if __name__ == "__main__":
    main()