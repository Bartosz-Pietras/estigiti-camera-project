import torch.onnx
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import onnx
from onnx_tf.backend import prepare
from tensorflow.keras.models import load_model
from model import Model


def evaluate_model(model, test_loader, epochs, train_size, valid_size, train_losses, validation_losses):
    criterion = nn.CrossEntropyLoss()
    device = select_proper_device()
    cf_matrix = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    accuracy, test_loss = 0, 0
    test_losses = np.array([])

    # Evaluate the validation loss and accuracy
    with torch.no_grad():
        test_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            results = model(images)
            batch_loss = criterion(results, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(results)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Reducing the dimensionality of the tensor
            top_class_cm = torch.squeeze(
                top_class, 1
            )
            cf_matrix_batch = confusion_matrix(
                labels.cpu().numpy(), top_class_cm.cpu().numpy()
            )
            cf_matrix = np.add(cf_matrix, cf_matrix_batch)

            test_losses = np.append(test_losses, test_loss)

        print(
            f"Test loss: {test_loss / len(test_loader):.3f}\n"
            f"Test accuracy: {accuracy / len(test_loader):.3f}"
        )

    plot_confusion_matrix(cf_matrix)
    plot_losses(epochs, train_losses, validation_losses, train_size, valid_size)


def plot_confusion_matrix(cf_matrix):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n');
    ax.set_xlabel('Predicted stage')
    ax.set_ylabel('Actual stage\n');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
    ax.yaxis.set_ticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])

    plt.show()


def plot_losses(epochs, train_losses, validation_losses, train_size, valid_size):
    fig, ax = plt.subplots()
    ax.set_xticks(epochs)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss value")
    ax.plot(epochs, train_losses / train_size, label="Training loss")
    ax.plot(epochs, validation_losses / valid_size, label="Validation loss")
    ax.legend(frameon=False)
    ax.grid()
    plt.show()


# Function to Convert to ONNX
def convert_to_ONNX():
    torch_model_path = "../model/leaves_recogniser.pth"
    onnx_model_path = "../model/leaves_recogniser.onnx"

    device = select_proper_device()

    model = Model()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(torch_model_path))
    else:
        model.load_state_dict(torch.load(torch_model_path, map_location="cpu"))

    # set the model to inference mode
    model.eval()

    dummy_input = torch.randn(
        5, 3, 298, 224
    )  # 5 batches (train.py), 3 channels on image, dimensions of the image

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs) - na razie dalem None bo nie wiem do konca
        onnx_model_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
    )

    print("\nModel has been converted to ONNX")


def onnx_to_tf_weights():
    """Saving the weights instead of a model in tensorflow format, since that's what our model (based on CIFAR) needs."""
    onnx_model_path = "../model/leaves_recogniser.onnx"
    tensorflow_model_path = "../model/leaves_recogniser.pb"
    tensorflow_weights_path = "../model/leaves_recogniser_weights"

    onnx_model = torch.onnx.load(onnx_model_path)

    tf_model_save = prepare(onnx_model)
    tf_model_save.export_graph(tensorflow_model_path)

    tf_model = load_model(tensorflow_model_path)
    tf_model.save_weights(tensorflow_weights_path)


def select_proper_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    onnx_to_tf_weights()
