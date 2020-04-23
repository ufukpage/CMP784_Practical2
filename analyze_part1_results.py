
import torch, os, sys
import matplotlib.pyplot as plt


def plot_history(training_data, validation_data, title="Training Time", y_label="Accuracy",
                 fig_name="dummy.png", show=True):
    epochs = list(range(0, len(training_data), 1))

    plt.plot(epochs, training_data, 'b', label='Training '+y_label)
    plt.plot(epochs, validation_data, 'r', label='Val '+ y_label)
    # b is for "solid blue line"
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()

    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    figs_dir = os.path.join(current_path, 'part1_results')
    os.makedirs(figs_dir, exist_ok=True)

    plt.savefig(os.path.join(figs_dir, fig_name ), bbox_inches='tight')
    if show:
        plt.show()


def write_params(model_name, checkpoint):

    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    figs_dir = os.path.join(current_path, 'part1_results')
    os.makedirs(figs_dir, exist_ok=True)

    filepath = os.path.join(figs_dir, model_name+'.txt')
    file = open(filepath, 'w')
    # Writing a string to file

    for key in checkpoint:
        file.write(str(key)+":\t"+str(checkpoint[key]) + "\n")

    # Closing file
    file.close()

directory = "part1_models"

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

        fig_name = model_name + "_train.png"
        plot_history(train_accs, val_accs, title="Training Time", y_label="Accuracy", fig_name=model_name+"_acc.png")
        plot_history(train_losses, val_losses, title="Training Time", y_label="Loss", fig_name=model_name+"_loss.png")
        write_params(model_name, checkpoint)

