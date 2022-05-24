import torch.onnx
from model import Model

# Function to Convert to ONNX
def convert_to_ONNX():
    torch_model_path = "../model/leaves_recogniser.pth"
    onnx_model_path = "../model/leaves_recogniser.onnx"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(torch_model_path))
    else:
        model.load_state_dict(torch.load(torch_model_path, map_location='cpu'))

    # set the model to inference mode
    model.eval()

    dummy_input = torch.randn(5, 3, 298, 224) # 5 batches (train.py), 3 channels on image, dimensions of the image

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs) - na razie dalem None bo nie wiem do konca
                      onnx_model_path,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      )

    print('\nModel has been converted to ONNX')

if __name__ == "__main__":
    convert_to_ONNX()

