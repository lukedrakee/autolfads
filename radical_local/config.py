"""
RADICaL Configuration for 2-Photon Calcium Imaging

This is the ONLY file you need to edit to configure training.
"""

# =============================================================================
# PATHS - Edit these to match your setup
# =============================================================================
DATA_DIR = "./data"                    # Where your HDF5 data files are
OUTPUT_DIR = "./outputs"               # Where model checkpoints and results go
TRAIN_FILE = "train_data.h5"           # Training data filename
VALID_FILE = "valid_data.h5"           # Validation data filename

# =============================================================================
# RADICaL-SPECIFIC SETTINGS (What makes it work for calcium imaging)
# =============================================================================
OUTPUT_DIST = "zi-gamma"               # Zero-inflated gamma (DO NOT CHANGE for calcium)
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
