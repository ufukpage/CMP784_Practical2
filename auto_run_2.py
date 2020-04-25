from run_part2 import run

MODEL_SAVE_PATH = "model1_part2.tar"
LR = 1e-3
Momentum = 0.9
BATCH_SIZE = 1024
USE_CUDA = True
NUM_EPOCHS = 200
PATIENCE = -1
ALEXNET_EXTRACTOR = False
FINETUNE_ALEXNET = False
VISUALIZE = False
LR_SCHEDULER_STEP = 10
MODEL_LOAD_PATH = MODEL_SAVE_PATH

if __name__ == "__main__":
    run(model_save_path=MODEL_SAVE_PATH, LR=LR, momentum=0.9, batch_size=BATCH_SIZE, alex_net_extractor=ALEXNET_EXTRACTOR,
        fine_tune_alexnet=FINETUNE_ALEXNET, visualize=VISUALIZE, num_epochs=NUM_EPOCHS, patience=PATIENCE,
        scheduler_step=LR_SCHEDULER_STEP, model_load_path=MODEL_LOAD_PATH)