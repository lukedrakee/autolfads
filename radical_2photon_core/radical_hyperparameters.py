"""
RADICaL Hyperparameters for 2-Photon Calcium Imaging
=====================================================

This file contains the essential hyperparameters needed to configure LFADS
for 2-photon calcium imaging data (RADICaL = Rate and Dynamics Inferred from Calcium).

The key difference from standard LFADS (for spike data) is:
- output_dist = 'zi-gamma' instead of 'poisson'
- Additional parameters for calcium dynamics (temporal_shift, gamma_prior, etc.)

Usage:
------
These hyperparameters are passed to the LFADS model (from lfadslite package).
Adapt this configuration to your training script/framework.
"""

# =============================================================================
# CRITICAL: This parameter makes it RADICaL (for calcium) vs LFADS (for spikes)
# =============================================================================
OUTPUT_DISTRIBUTION = 'zi-gamma'  # Zero-inflated gamma for 2-photon calcium data
                                  # Use 'poisson' for spike count data

# =============================================================================
# RADICaL-SPECIFIC HYPERPARAMETERS (for calcium imaging)
# =============================================================================
RADICAL_PARAMS = {
    # Output distribution for calcium fluorescence
    'output_dist': 'zi-gamma',

    # Temporal shift - accounts for calcium indicator dynamics
    # Set to 0 for no shift, or tune based on your indicator (GCaMP6f, etc.)
    'temporal_shift': 0,

    # Distribution for temporal shift sampling
    'temporal_shift_dist': 'normal',

    # Transform from latent factors to firing rates
    # 'linscaledsigmoid' works well for calcium imaging
    'fac_2_rates_transform': 'linscaledsigmoid',

    # Whether to apply temporal shift during posterior sampling
    'apply_temporal_shift_during_posterior_sampling': False,

    # Log transform input (typically False for dF/F data)
    'log_transform_input': False,

    # Feedforward keep probability (dropout)
    'ff_keep_prob': 1.0,

    # L2 regularization on factor-to-rates transformation
    'l2_fac_2_rates_scale': 1e-3,  # Search range: (1e-6, 1.0)

    # Gamma distribution prior - IMPORTANT for calcium imaging
    # Controls the shape of the inferred rate distribution
    'gamma_prior': 20.0,  # Search range: (1.0, 100.0), explorable=True

    # L2 penalty on gamma distance
    'l2_gamma_distance_scale': 1e-4,

    # Minimum scale parameter for gamma distribution
    's_min': 0.1,
}

# =============================================================================
# LFADS ARCHITECTURE HYPERPARAMETERS
# =============================================================================
ARCHITECTURE_PARAMS = {
    # Latent factors dimension - number of latent dimensions to infer
    # Tune based on expected complexity of neural dynamics
    'factors_dim': 40,
    'in_factors_dim': 0,

    # Generator RNN - generates neural dynamics
    'gen_dim': 64,

    # Initial condition encoder - encodes trial start state
    'ic_dim': 64,
    'ic_enc_dim': 64,
    'ic_enc_seg_len': 0,  # 0 = non-causal encoder

    # Controller - models external inputs/perturbations
    'co_dim': 2,              # Controller output dimension
    'ci_enc_dim': 64,         # Controller input encoder dimension
    'con_dim': 64,            # Controller dimension
    'do_causal_controller': False,
    'controller_input_lag': 1,

    # External inputs (set to 0 if no external inputs)
    'ext_input_dim': 0,
}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
TRAINING_PARAMS = {
    # Learning rate
    'learning_rate_init': 0.001,  # Search range: (0.00001, 0.005)

    # Batch sizes
    'batch_size': 480,
    'valid_batch_size': 4000,

    # Validation metric
    'val_cost_for_pbt': 'heldout_trial',  # or 'heldout_samp' for sample validation

    # Regularization - Dropout
    'keep_prob': 0.95,      # Search range: (0.3, 1.0)
    'keep_ratio': 0.5,      # Coordinated dropout, search range: (0.01, 0.99)
    'cv_keep_ratio': 1.0,
    'cd_grad_passthru_prob': 0.0,

    # L2 Regularization
    'l2_gen_scale': 1e-3,       # Generator L2, search range: (1e-5, 1e-1)
    'l2_ic_enc_scale': 0.0,     # IC encoder L2
    'l2_con_scale': 1e-3,       # Controller L2, search range: (1e-5, 1e-1)
    'l2_ci_enc_scale': 0.0,     # CI encoder L2

    # KL divergence weights
    'kl_co_weight': 1e-5,       # Controller KL, search range: (1e-6, 1e-4)
    'kl_ic_weight': 1e-5,       # Initial condition KL, search range: (1e-6, 1e-4)

    # KL/L2 ramping schedule
    'kl_start_epoch': 0,
    'l2_start_epoch': 0,
    'kl_increase_epochs': 80,
    'l2_increase_epochs': 80,

    # Optimizer
    'adam_epsilon': 1e-8,
    'beta1': 0.9,
    'beta2': 0.999,
    'loss_scale': 1e4,

    # Learning rate decay (set to 1 for no decay)
    'learning_rate_decay_factor': 1,
    'learning_rate_stop': 1e-10,
}

# =============================================================================
# DATA FORMAT PARAMETERS
# =============================================================================
DATA_PARAMS = {
    'data_filename_stem': 'lfads',  # Data files must start with this prefix
    'do_train_readin': False,
    'do_train_encoder_only': False,
    'cv_rand_seed': 1000,
}


def get_all_hyperparameters():
    """Return all hyperparameters as a single dictionary."""
    all_params = {}
    all_params.update(RADICAL_PARAMS)
    all_params.update(ARCHITECTURE_PARAMS)
    all_params.update(TRAINING_PARAMS)
    all_params.update(DATA_PARAMS)
    return all_params


def print_radical_config():
    """Print the RADICaL-specific configuration."""
    print("=" * 60)
    print("RADICaL Configuration for 2-Photon Calcium Imaging")
    print("=" * 60)
    print(f"\nOutput Distribution: {OUTPUT_DISTRIBUTION}")
    print("\nRADICaL-Specific Parameters:")
    for key, value in RADICAL_PARAMS.items():
        print(f"  {key}: {value}")
    print("\nArchitecture Parameters:")
    for key, value in ARCHITECTURE_PARAMS.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print_radical_config()
