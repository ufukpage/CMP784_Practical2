import torch, os, sys
import matplotlib.pyplot as plt


def plot_history(training_data, validation_data, title="Training Time", y_label="Accuracy",
                 fig_name="dummy.png", show=True, write_path="part1_results", isLoss = False):
    epochs = list(range(0, len(training_data), 1))
    axes = plt.gca()
    if not isLoss:
        axes.set_ylim([0,1])
    plt.plot(epochs, training_data, 'b', label='Training ' + y_label)
    plt.plot(epochs, validation_data, 'r', label='Val ' + y_label)
    # b is for "solid blue line"
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()

    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    figs_dir = os.path.join(current_path, write_path)
    os.makedirs(figs_dir, exist_ok=True)

    plt.savefig(os.path.join(figs_dir, fig_name ), bbox_inches='tight')
    if show:
        plt.show()


def write_params(model_name, checkpoint, write_path="part1_results"):

    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    figs_dir = os.path.join(current_path, write_path)
    os.makedirs(figs_dir, exist_ok=True)

    filepath = os.path.join(figs_dir, model_name+'.txt')
    file = open(filepath, 'w')

    for key in checkpoint:
        file.write(str(key)+":\t"+str(checkpoint[key]) + "\n")

    file.close()


def analyze_part(directory="part1_models", write_path="part1_results"):
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(directory):
        for file in files:
            model_name = file.split(".")[0]
            model_load_path = model_name+".tar"

            checkpoint = torch.load(os.path.join(directory, model_load_path))

            train_accs = checkpoint["history"]["training_acc"]
            train_losses = checkpoint["history"]["training_loss"]
            val_accs = checkpoint["history"]["val_acc"]
            val_losses = checkpoint["history"]["val_loss"]

            plot_history(train_accs, val_accs, title="Training Time", y_label="Accuracy",
                         fig_name=model_name+"_acc.png", write_path=write_path)
            plot_history(train_losses, val_losses, title="Training Time", y_label="Loss",
                         fig_name=model_name+"_loss.png", isLoss=True, write_path=write_path)
            write_params(model_name, checkpoint, write_path=write_path)


if __name__ == "__main__":
    # analyze_part(directory="part1_models", write_path="part1_results")
    # analyze_part(directory="part2_models", write_path="part2_results")

    model_path = os.path.join("part2_models", "6-model1_part2_best.tar")
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint["model_state_dict"]
    weights = model_state_dict["hidden_fc_layer.weight"].cpu().numpy()

    weights[0]

    print(weights.shape)