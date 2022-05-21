import torch
from model import Model
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((298, 224)),
                                transforms.PILToTensor(),
                                ])
model = Model()
model.load_state_dict(torch.load("../model/leaves_recogniser.pth"))

test_image_with_leaves = Image.open("../dataset/with_leaves/IMG_6835.jpeg")
test_image_no_leaves = Image.open("../dataset/no_leaves/IMG_6395.jpeg")

test_image_with_leaves = test_image_with_leaves.rotate(-90, Image.NEAREST, expand=True)
test_image_no_leaves = test_image_no_leaves.rotate(-90, Image.NEAREST, expand=True)

test_1 = transform(test_image_with_leaves)
test_2 = transform(test_image_no_leaves)


output_1 = torch.max(model(test_1.float()), 1).indices
output_2 = torch.max(model(test_2.float()), 1).indices

outputs = [output_1, output_2]

for i, output in enumerate(outputs):
    if output.item():
        print(f"There are leaves on the {i+1}. image.")
    else:
        print(f"There are no leaves on the {i+1}. image.")

test_image_with_leaves.show()
test_image_no_leaves.show()
