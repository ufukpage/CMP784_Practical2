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

            # torch.nn.Module.dump_patches = True
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


def visualize_weights(model_path, channel=0, visualize_weights_folder_name="visual_weights", show=False):
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    figs_dir = os.path.join(current_path, visualize_weights_folder_name)
    os.makedirs(figs_dir, exist_ok=True)

    checkpoint = torch.load(model_path)

    model_state_dict = checkpoint["model_state_dict"]

    weights = model_state_dict["hidden_fc_layer.weight"]
    for index, weight in enumerate(weights):

        neuron_weights = weight.view(-1, 28, 28)
        neuron_weights = neuron_weights[channel, : , :]
        n_w = neuron_weights.cpu().numpy()

        w = plt.imshow(n_w, interpolation='nearest')
        w.set_cmap("RdBu")
        plt.colorbar(w, orientation='vertical')

        fig_path = os.path.join(figs_dir, str(index)+".png")
        if os.path.isfile(fig_path):
            os.remove(fig_path)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close("all")
        if show:
            plt.show()


if __name__ == "__main__":
    # analyze_part(directory="part1_models", write_path="part1_results")
    # analyze_part(directory="part2_models", write_path="part2_results")

    # model_path = os.path.join("part2_models", "model1_part2_800.tar")
    # visualize_weights(model_path, show=False, visualize_weights_folder_name="weights_800", channel=0)
    # visualize_weights(model_path, show=False, visualize_weights_folder_name="weights_801", channel=1)
    # visualize_weights(model_path, show=False, visualize_weights_folder_name="weights_802", channel=2)

    import gradient_visualization.guided_backprop as gb
    # import gradient_visualization.gradcam as gradcam
    from PIL import Image

    from gradient_visualization.misc_functions import preprocess_image, save_gradient_images

    model_path = os.path.join("part2_models", "alexnet.tar")
    checkpoint = torch.load(model_path)
    torch.nn.Module.dump_patches = True
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['model_state_dict'])

    image = Image.open("statham_ornek.jpg").convert('RGB')

    image = preprocess_image(image)
    dataset_dir = "dataset_part2/train"
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    target_class = classes.index("Jason_Statham")

    model_alexnet = model.alexnet.cuda()
    GBP = gb.GuidedBackprop(model_alexnet)

    guided_grads = GBP.generate_gradients(image.cuda(), target_class)
    file_name_to_export = "alexnet"
    save_gradient_images(guided_grads,  file_name_to_export + '_Guided_BP_color')

