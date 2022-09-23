#include "include/crow_all.h"
#include "include/IPManager.h"


using namespace std;
using namespace boost::interprocess;
int main() {
    // Create SharedMemory Buffer to share between main process and launched inference processes 
    char sm_file[50]="sharedBuf";
    shared_memory_object shm (open_or_create, sm_file, read_write);
    shm.truncate(NIP_MAX*MEM_BLOCK);
    mapped_region region(shm, read_write);
    void *sharedMemAddr= region.get_address();
    memset(sharedMemAddr,0, region.get_size());

    // Read label for classification
    std::vector<std::string> labels;
    std::string label;
    std::ifstream labelsfile ("../labels.txt");
    if (labelsfile.is_open())
    {
        while (getline(labelsfile, label))
        {
            labels.push_back(label);
        }
        labelsfile.close();
    }
    else{
        cout<<"Labels file not found"<<endl;
        return EXIT_FAILURE;
    }

    // Create Inference Process Manager
    IPManager IPMgr;
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
            string base64_image = args["image"].s();
            string model_name = args["model"].s();
            int SLO = args["slo"].i();
            cout<<"Received request for "<<model_name<<","<<SLO<<endl;
            int pred=IPMgr.handle(sharedMemAddr,base64_image,model_name,SLO);
            if (pred!=-1){
                result["Prediction"] = labels[pred];
                result["Status"] = "Success";
                os << crow::json::dump(result);
                return crow::response{os.str()};
            }
            else{
                os << crow::json::dump(result);
                return crow::response{os.str()};
            }
        } catch (std::exception& e){
        os << crow::json::dump(result);
        return crow::response(os.str());
        }

    });
    app.port(PORT).run();
    return EXIT_SUCCESS;
}