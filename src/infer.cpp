#include <torch/script.h>
#include <iostream>
#include <memory>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <ATen/cuda/CUDAGraph.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <cstring>
#include <cstdlib>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/cuda.h>
#include <stdio.h>
#include "consolelog.hpp"

#define MEM_BLOCK 1000000
using namespace boost::interprocess;


std::vector<float> get_outputs(torch::Tensor output) {
  int ndim = output.ndimension();
  assert(ndim == 2);

  torch::ArrayRef<int64_t> sizes = output.sizes();
  int n_samples = sizes[0];
  int n_classes = sizes[1];

  assert(n_samples == 1);

  std::vector<float> probs(output.data_ptr<float>(),
                                  output.data_ptr<float>() + (n_samples * n_classes));
  return probs;
}

/*
  Inference Process: ./infer model_name ip_id GPUid 
*/
int main(int argc, const char* argv[]) {
  //Open shared memory object and map the pre-defined shared memory area
  char sm_file[50]="sharedBuf";
  shared_memory_object shm (open_only, sm_file, read_write);
  mapped_region region(shm, read_write,MEM_BLOCK*atoi(argv[2]),MEM_BLOCK);
  // Load model
  std::unordered_map<std::string,std::string> model_path{
    {"resnet18","../model_dir/resnet18.pt"},
    {"resnet50","../model_dir/resnet50.pt"},
    {"vgg16","../model_dir/vgg16.pt"},
    {"vgg19","../model_dir/vgg19.pt"}
  };

  int GPUid=atoi(argv[3]);
  std::string device_string="cuda:"+std::to_string(GPUid);
  c10::InferenceMode guard;
  // get a new stream from CUDA stream pool on device
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, GPUid);
  // set the current CUDA stream to `myStream`
  at::cuda::setCurrentCUDAStream(myStream); 
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(model_path[argv[1]]);
  }
  catch (const c10::Error& e) {
    console.error("error loading the model");
    return -1;
  }
  console.info("Load model success");
  model.to(device_string);

  /* TODO: Speedup inference with CUDA Graph Stream capture, available since torch 1.10 */
  // auto g=at::cuda::CUDAGraph();
  // g.capture_begin();
  // static_out=model.forward({static_inp}).toTensor();
  // g.capture_end();
  
  int image_height = 224;
  int image_width = 224; 
  std::vector<int64_t> dims = {1, image_height, image_width, 3};
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};
  // CUDA Timer
  auto start=at::cuda::CUDAEvent(true);
  auto end=at::cuda::CUDAEvent(true);
  char *mem = static_cast<char*>(region.get_address());
  // While loop to check coming input from shared memory
  while (1){     
    if (*mem==0)
        continue;
    else{      
        // Read Image data from shared memory to tensor
        torch::Tensor input_tensor;
        torch::TensorOptions options(torch::kFloat32);
        input_tensor = torch::from_blob(mem+1, torch::IntList(dims), options);
        input_tensor = input_tensor.permute(torch::IntList(permute_dims));
        input_tensor = input_tensor.toType(torch::kFloat32);
        // start.record();
        torch::Tensor output = model.forward({input_tensor.to(device_string)}).toTensor();
        // end.record();
        // torch::cuda::synchronize();
        // cout<<"Inference time:"<<start.elapsed_time(end)<<"ms \n";
        std::vector<float> probs = get_outputs(output.to(at::kCPU));
        int pred=std::distance(probs.begin(),std::max_element(probs.begin(), probs.end()));
        // Write result to sharedBuf
        memset(mem+1,pred & 0xff,1);
        memset(mem+2,(pred >> 8) & 0xff,1);
        memset(mem, 0, 1); // set marker byte to 0
    }
  }
}