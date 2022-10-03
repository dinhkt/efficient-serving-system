#include "TRTengine.h"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#define MEM_BLOCK 1000000

using namespace boost::interprocess;

//Inference Process: ./infer_trt model_name ip_id GPUid
int main(int argc, const char* argv[]) {
    //Open shared memory object and map the pre-defined shared memory area
    char sm_file[50]="sharedBuf";
    shared_memory_object shm (open_only, sm_file, read_write);
    mapped_region region(shm, read_write,MEM_BLOCK*atoi(argv[2]),MEM_BLOCK);

    // Must specify a dynamic batch size when exporting the model to onnx.
    std::unordered_map<std::string,std::string> model_path{
        {"resnet18","../model_dir/resnet18.onnx"},
        {"resnet50","../model_dir/resnet50.onnx"},
        {"vgg16","../model_dir/vgg16.onnx"},
        {"vgg19","../model_dir/vgg19.onnx"}
    };

    Options options;
    options.optBatchSizes = {1, 2, 4, 8};
    options.deviceIndex=atoi(argv[3]);
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
    char *mem = static_cast<char*>(region.get_address());
    // While loop to check coming input from shared memory
    while (1){     
        if (*mem==0)
            continue;
        else{
            int bs=*mem;      
            engine.runInference(static_cast<void*>(mem)+1,bs,outputs);
            // now input only have bs=1
            int pred=std::distance(outputs[0].begin(),std::max_element(outputs[0].begin(), outputs[0].end()));
            // Write result to sharedBuf
            memset(mem+1,pred & 0xff,1);
            memset(mem+2,(pred >> 8) & 0xff,1);
            memset(mem, 0, 1); // set marker byte to 0
        }
    }
    return 0;
}
