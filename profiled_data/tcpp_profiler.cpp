#include <torch/script.h>
#include <iostream>
#include <memory>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <ATen/cuda/CUDAGraph.h>
#include <cstring>
#include <cstdlib>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/cuda.h>
#include <stdio.h>
#include "consolelog.hpp"

#define MEM_BLOCK 10000000
std::string MODEL_DIR("../../model_dir/");

/*
  Inference Process: ./infer model_name pGPU
*/
int main(int argc, const char* argv[]) {
  // Load model
  console.info(argv[0],argv[1],argv[2]);
  std::unordered_map<std::string,std::string> model_path;
  std::ifstream  data(MODEL_DIR+std::string("config_tcpp.txt"));
  std::string row;
  while (getline(data,row)){
    std::stringstream X(row);
    std::string arr[2];
    int i=0;
    while (getline(X,arr[i],',')){
        i+=1;
        if (i==2){
            model_path[arr[0]]=MODEL_DIR+arr[1];
            i=0;
        }
    }
  }
  
  int pGPU=atoi(argv[2]);
  
  std::string device_string="cuda:0";
  c10::InferenceMode guard;
  // get a new stream from CUDA stream pool on device
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, 0);
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
  
  std::ofstream myfile;
  myfile.open ("../tcpp_profiler.txt",std::ios_base::app);
  auto start=at::cuda::CUDAEvent(true);
  auto end=at::cuda::CUDAEvent(true);
  void *mem=malloc(10000000);
  memset(mem,1,10000000);
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};
  for (int bs=1;bs<9;bs++){     
        // Warmup      
        for(int i=0;i<10;i++)
        {
            std::vector<int64_t> dims = {bs, 224, 224, 3};
            torch::Tensor inp;
            torch::TensorOptions options(torch::kFloat32);
            inp = torch::from_blob(mem, torch::IntList(dims), options);
            inp = inp.permute(torch::IntList(permute_dims));
            inp = inp.toType(torch::kFloat32);
            at::Tensor out=model.forward({inp.to(device_string)}).toTensor();
        }
        float total_time=0;

        // torch::Tensor input_tensor = torch::randn({bs,3,224,224},torch::dtype(torch::kFloat32).device(torch::kCPU));
        for (int i=0;i<20;i++){
            std::vector<int64_t> dims = {bs, 224, 224, 3};
            torch::Tensor input_tensor;
            torch::TensorOptions options(torch::kFloat32);
            input_tensor = torch::from_blob(mem, torch::IntList(dims), options);
            input_tensor = input_tensor.permute(torch::IntList(permute_dims));
            input_tensor = input_tensor.toType(torch::kFloat32);
            start.record();
            torch::Tensor output = model.forward({input_tensor.to(device_string)}).toTensor();
            end.record();
            torch::cuda::synchronize();
            total_time+=start.elapsed_time(end);
            // console.info(start.elapsed_time(end));
        }
        myfile<<argv[1]<<","<<pGPU<<","<<bs<<","<<total_time/20<<std::endl;
        console.info(argv[1],pGPU,bs,"Inference time:",total_time/20,"ms");
  }
}
