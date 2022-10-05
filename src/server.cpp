#include "crow.h"
#include "IPManager.h"
#include "consolelog.hpp"
#include<signal.h>

using namespace boost::interprocess;
#define PORT 8082

void sig_handler(int signum){
    shared_memory_object::remove("sharedBuf");
}

void readImageNetlabels(std::vector<std::string> &labels,const char* labels_path){
    std::string label;
    std::ifstream labelsfile (labels_path);
    if (labelsfile.is_open())
    {
        while (getline(labelsfile, label))
        {
            labels.push_back(label);
        }
        labelsfile.close();
    }
}

int main(int argc, const char* argv[]) {
    signal(SIGINT,sig_handler);
    signal(SIGQUIT,sig_handler);
    signal(SIGTERM,sig_handler);
    signal(SIGKILL,sig_handler);
    // Read label for image classification
    std::vector<std::string> labels;
    readImageNetlabels(labels,"../labels.txt");
    if (labels.size()==0){
        console.error("Labels file not found");
        return EXIT_FAILURE;
    }
    // Create SharedMemory Buffer to share between main process and launched inference processes 
    shared_memory_object shm (open_or_create, "sharedBuf", read_write);
    shm.truncate(NIP_MAX*MEM_BLOCK);
    mapped_region region(shm, read_write);
    void *sharedMemAddr= region.get_address();
    memset(sharedMemAddr,0, region.get_size());
    // Create Inference Process Manager
    IPManager IPMgr;
    if (argc==2 && strcmp(argv[1],"tcpp")==0)
        IPMgr.setInferType(0);
    else if (argc==2 && strcmp(argv[1],"trt")==0)
        IPMgr.setInferType(1);
    else{
        console.error("Usage: ./server <infer_mode>\ninfer_mode=tcpp for torch c++ \ninfer_mode=trt for tensorrt ");
        return EXIT_FAILURE;
    }
    IPMgr.run();
    //REST API by Crow 
    crow::SimpleApp app;
    CROW_ROUTE(app, "/predict").methods("POST"_method, "GET"_method)
    ([sharedMemAddr,&IPMgr,&labels](const crow::request& req){
        crow::json::wvalue result;
        result["Prediction"] = "";
        result["Status"] = "Failed";
        std::ostringstream os;

        try {
            auto args = crow::json::load(req.body);
            std::string base64_image = args["image"].s();
            std::string model_name = args["model"].s();
            int SLO = args["slo"].i();
            console.info("Received request for",model_name,",slo =",SLO);
            int pred=IPMgr.handle(sharedMemAddr,base64_image,model_name,SLO);
            if (pred!=-1){
                result["Prediction"] = labels[pred];
                result["Status"] = "Success";
                return crow::response{result.dump()};
            }
            else{
                return crow::response{result.dump()};
            }
        } catch (std::exception& e){
        return crow::response(result.dump());
        }

    });
    app.port(PORT).multithreaded().run();
    return EXIT_SUCCESS;
    
}