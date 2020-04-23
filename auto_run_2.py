from run_part2 import run

MODEL_SAVE_PATH = "model1_part2.tar"
LR = 1e-3
Momentum = 0.9 # If you use SGD with momentum
BATCH_SIZE = 16
USE_CUDA = True
NUM_EPOCHS = 1
PATIENCE = 5
ALEXNET_EXTRACTOR = False
FINETUNE_ALEXNET = False
VISUALIZE = True

if __name__ == "__main__":
    run(MODEL_SAVE_PATH, LR, Momentum, BATCH_SIZE, POOLING, NUM_EPOCHS, PATIENCE, DATA_PATH, MODEL_LOAD_PATH)