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
#include "include/imageutils.h"
#include "include/base64.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MEM_BLOCK 1000000
using namespace std;
using namespace boost::interprocess;
using namespace cv;

/*
  Inference Process: ./infer model_name ip_id GPUid 
*/
int main(int argc, const char* argv[]) {
  //Open shared memory object and map the pre-defined shared memory area
  char sm_file[50]="sharedBuf";
  shared_memory_object shm (open_only, sm_file, read_write);
  mapped_region region(shm, read_write,MEM_BLOCK*atoi(argv[2]),MEM_BLOCK);
  // Load model
  unordered_map<string,string> model_path{
    {"resnet18","../model_dir/resnet18.pt"},
    {"resnet50","../model_dir/resnet50.pt"},
    {"vgg16","../model_dir/vgg16.pt"},
    {"vgg19","../model_dir/vgg19.pt"}
  };

  int GPUid=atoi(argv[3]);
  string device_string="cuda:"+to_string(GPUid);
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
    cerr << "error loading the model\n";
    return -1;
  }
  cout << "Load model success\n";
  model.to(device_string);

  /* TODO: Speedup inference with CUDA Graph Stream capture, available since torch 1.10 */
  // auto g=at::cuda::CUDAGraph();
  // g.capture_begin();
  // static_out=model.forward({static_inp}).toTensor();
  // g.capture_end();
  
  // Preprocess params
  int image_height = 224;
  int image_width = 224; 
  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> std = {0.229, 0.224, 0.225};
  // CUDA Timer
  auto start=at::cuda::CUDAEvent(true);
  auto end=at::cuda::CUDAEvent(true);
  char *mem = static_cast<char*>(region.get_address());
  // While loop to check coming input from shared memory
  while (1){     
    if (*mem==0)
        continue;
    else{      
        // Read Image data from shared memory
        std::string base64_image(mem,MEM_BLOCK);
        std::string decoded_image = base64_decode(base64_image);
        std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
        cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);
        // Preprocess image
        image = preprocess(image, image_height, image_width, mean, std);
        // Forward
        torch::Tensor tensor = convert_images_to_tensor({image,});
        torch::Tensor output = model.forward({tensor.to(device_string)}).toTensor();
        std::vector<float> probs = get_outputs(output.to(at::kCPU));
        // Postprocess
        int pred;
        float prob;
        tie(pred, prob) = postprocess(probs);
        // Write result to sharedBuf
        memset(mem+1,pred & 0xff,1);
        memset(mem+2,(pred >> 8) & 0xff,1);
        memset(mem, 0, 1); // set marker byte to 0
    }
  }
}
