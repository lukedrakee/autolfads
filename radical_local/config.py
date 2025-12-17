"""
RADICaL Configuration for 2-Photon Calcium Imaging

This is the ONLY file you need to edit to configure training.
"""

# =============================================================================
# PATHS - Edit these to match your setup
# =============================================================================
DATA_DIR = "./data"                    # Where your HDF5 data files are
OUTPUT_DIR = "./outputs"               # Where model checkpoints and results go
DATA_FILE = "data.h5"                  # Your data filename (single file)

# =============================================================================
# TRAIN/VALIDATION SPLIT
# =============================================================================
TRAIN_SPLIT = 0.8                      # Fraction of data for training (0.8 = 80% train, 20% valid)

# =============================================================================
# DATA FORMAT - Choose based on your preprocessing
# =============================================================================
# Options:
#   "zi-gamma"  - For raw or dF/F calcium (non-negative values only)
#   "gaussian"  - For z-scored data (can have negative values)
#
# If your data is z-scored, use "gaussian"
# If your data is raw fluorescence or dF/F, use "zi-gamma"
OUTPUT_DIST = "gaussian"               # Use "gaussian" for z-scored data

# Only used if OUTPUT_DIST = "zi-gamma"
GAMMA_PRIOR = 20.0                     # Prior on gamma shape (tune: 1-100)
S_MIN = 0.1                            # Minimum scale parameter

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
FACTORS_DIM = 40                       # Latent factor dimensions (tune: 10-100)
GEN_DIM = 64                           # Generator RNN hidden size
IC_DIM = 64                            # Initial condition dimensions
CO_DIM = 2                             # Controller output dimensions (0 to disable)

# =============================================================================
# TRAINING
# =============================================================================
BATCH_SIZE = 64                        # Reduce if GPU runs out of memory
LEARNING_RATE = 0.001                  # Initial learning rate
MAX_EPOCHS = 500                       # Total training epochs
DROPOUT_KEEP = 0.95                    # Dropout keep probability

# =============================================================================
# REGULARIZATION
# =============================================================================
L2_GEN = 1e-3                          # L2 on generator weights
L2_CON = 1e-3                          # L2 on controller weights
KL_IC_WEIGHT = 1e-5                    # KL weight on initial conditions
KL_CO_WEIGHT = 1e-5                    # KL weight on controller outputs
