from run import run

MODEL_SAVE_PATH = "model1.tar"
LR = 1e-3
Momentum = 0.9 # If you use SGD with momentum
BATCH_SIZE = 512
USE_CUDA = True
POOLING = True
NUM_EPOCHS = 200
PATIENCE = -1
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.2
NUM_ARTISTS = 11
DATA_PATH = "./art_data/artists"
MODEL_LOAD_PATH = MODEL_SAVE_PATH


if __name__ == "__main__":
    run(MODEL_SAVE_PATH, LR, Momentum, BATCH_SIZE, POOLING, NUM_EPOCHS, PATIENCE, DATA_PATH, MODEL_LOAD_PATH)