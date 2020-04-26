from run_part2 import run

MODEL_SAVE_PATH = "model1_part2.tar"
LR = 2e-3
Momentum = 0.9
BATCH_SIZE = 8192
NUM_EPOCHS = 500
PATIENCE = 20
ALEXNET_EXTRACTOR = True
FINETUNE_ALEXNET = True
VISUALIZE = False
LR_SCHEDULER_STEP = 60
MODEL_LOAD_PATH = MODEL_SAVE_PATH
WD = 1e-2
if __name__ == "__main__":
    run(model_save_path=MODEL_SAVE_PATH, LR=LR, momentum=0.9, batch_size=BATCH_SIZE, alex_net_extractor=ALEXNET_EXTRACTOR,
        fine_tune_alexnet=FINETUNE_ALEXNET, visualize=VISUALIZE, num_epochs=NUM_EPOCHS, patience=PATIENCE,
        scheduler_step=LR_SCHEDULER_STEP, model_load_path=MODEL_LOAD_PATH, weight_decay=WD)
