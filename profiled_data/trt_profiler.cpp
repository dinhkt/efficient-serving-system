#include "TRTengine.h"
#include "consolelog.hpp"

std::string MODEL_DIR("../../model_dir/");
int main(int argc, const char* argv[]) {
    // Must specify a dynamic batch size when exporting the model to onnx.
    console.info("Created",argv[1],argv[2]);
    std::unordered_map<std::string,std::string> model_path;
    std::ifstream  data(MODEL_DIR+std::string("config_trt.txt"));
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
    Options options;
    options.optBatchSizes = {1, 2, 3,4,5,6,7,8};
    options.deviceIndex=0;
    std::string model_name(argv[1]);
    options.model_name=model_name;
    TRTengine engine(options);

    const std::string onnxModelpath = model_path[argv[1]];
    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    std::vector<std::vector<float>> outputs;
    void *mem=malloc(10000000);
    memset(mem,1,10000000);

    std::ofstream myfile;
    myfile.open ("../trt_profiler.txt",std::ios_base::app);
    for (int bs=1;bs<9;bs++){     
        // Warmup      
        for(int i=0;i<10;i++)
        {   
            engine.runInference(static_cast<void*>(mem)+1,bs,outputs);
        }
        float total_time=0;

        // torch::Tensor input_tensor = torch::randn({bs,3,224,224},torch::dtype(torch::kFloat32).device(torch::kCPU));
        for (int i=0;i<20;i++){
            auto start = std::chrono::high_resolution_clock::now();
            engine.runInference(static_cast<void*>(mem)+1,bs,outputs);
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        }
        myfile<<argv[1]<<","<<pGPU<<","<<bs<<","<<total_time/20000<<std::endl;
        console.info(argv[1],pGPU,bs,"Inference time:",total_time/20000,"ms");
  }
}
