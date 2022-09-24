import torch
import torchvision
import torchvision.transforms as T
import cv2
import onnx
from torchvision.io import read_image

def preprocess(img_path):
    # transformations for the input data
    transforms = T.Compose([
        T.Resize([224, 224]),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    image=transforms(read_image(img_path).type(torch.FloatTensor))

    batch_data = torch.unsqueeze(image, 0)
    print(batch_data.shape)
    return batch_data


model=torchvision.models.resnet50(pretrained=True)
input=preprocess("../test_client/dog.jpg").cuda()
model.eval()
model.cuda()
out=model(input)

# torch.onnx.export(model,input,"resnet50.onnx",input_names=["input"],output_names=["output"],export_params=True)

onnx_model=onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)

