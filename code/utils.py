import torch.onnx
from model import Model

# Function to Convert to ONNX
def Convert_to_ONNX(path):

    model = Model()

    # set the model to inference mode
    model.eval()

    # Export the model
    torch.onnx.export(model,  # model being run
                      None,  # model input (or a tuple for multiple inputs) - na razie dalem None bo nie wiem do konca
                      path,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})

    print('\nModel has been converted to ONNX')