import torch
import torchvision
import onnx




model=torchvision.models.vgg19(pretrained=True)
model.eval()
scripted_model=torch.jit.script(model)

batch_size=1# random initialization

dummy_input = torch.randn(batch_size, 3, 224, 224) 
dynamic_axes = {'input' : {0 : 'batch_size'}, 
                            'output' : {0 : 'batch_size'}}
torch.onnx.export(model, dummy_input, "vgg19.onnx",
                  do_constant_folding=True, 
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],
                  dynamic_axes=dynamic_axes)


#torch.onnx.export(scripted_model,input,"resnet50.onnx",input_names=["input"],output_names=["output"],export_params=True)

onnx_model=onnx.load("vgg19.onnx")
onnx.checker.check_model(onnx_model)

