#include "TRTengine.h"
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#define MEM_BLOCK 1000000
using namespace boost::interprocess;
std::string MODEL_DIR("../model_dir/");

//Inference Process: ./infer_trt model_name ip_id GPUid
int main(int argc, const char* argv[]) {
    //Open shared memory object and map the pre-defined shared memory area
    char sm_file[50]="sharedBuf";
    shared_memory_object shm (open_only, sm_file, read_write);
    mapped_region region(shm, read_write,MEM_BLOCK*atoi(argv[2]),MEM_BLOCK);

    // Must specify a dynamic batch size when exporting the model to onnx.
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
            console.info("read",bs);
            engine.runInference(static_cast<void*>(mem)+1,bs,outputs);
            for (int i=0;i<bs;i++){
                int pred=std::distance(outputs[i].begin(),std::max_element(outputs[i].begin(), outputs[i].end()));
                console.info(pred);
                memset(mem+1+2*i,pred & 0xff,1);
                memset(mem+2+2*i,(pred >> 8) & 0xff,1);
            }
            memset(mem, 0, 1); // set marker byte to 0
        }
    }
    return 0;
}
