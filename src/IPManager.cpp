#include "IPManager.h"
#include "consolelog.hpp"

using namespace boost::interprocess;


void IPManager::run(){
    running=true;
    // info of GPUs resources available
    n_GPU=at::cuda::device_count();
    for (int i=0;i<n_GPU;i++){
        GPUresources.push_back(100);
    }
    console.info("System has",n_GPU,"GPU");
    boost::thread th(&IPManager::IPsTimer, this);
    
    std::string row;
    std::string fpath;
    if (InferType==0){
	    fpath="../profiled_data/tcpp_profiler.txt";
    }
    else{
	    fpath="../profiled_data/trt_profiler.txt";
    }
    std::ifstream  data(fpath); 
    if (data.is_open())
    {
        std::string tmp,name,p,bs,t;
        int i=0;
        while (getline(data,row)){
            std::stringstream X(row);
            std::string arr[4];
            while (getline(X,arr[i],',')){
                i+=1;
                if (i==4){
                    if (stoi(arr[2])==1)
                        profiler[arr[0]][stoi(arr[1])]=stof(arr[3]);
                    i=0;
                }
            }
        }
        data.close();
    }
    else{
        console.error("Profiler file not found");
        return;
    }
    
}
void IPManager::IPsTimer(){
    while (true){
        auto now=std::chrono::steady_clock::now();
        std::vector<int> expired_list;
        for(auto iter = ipidTimer.begin(); iter != ipidTimer.end(); ++iter){
            if (std::chrono::duration_cast<std::chrono::seconds>(now-iter->second).count()>time_out)
                expired_list.push_back(iter->first);
        }

        for (auto i: expired_list){
            auto p=find_if(ip_list.begin(),ip_list.end(),
                            [i](const InferenceProcess& ip) {return ip.ip_id==i;});
            std::string model=p->model;
            kill(p->pid,SIGTERM);
            GPUresources[p->allocatedGPU]+=p->pGPU;
            std::atomic<int> ip_pid;
            while ((ip_pid = waitpid(p->pid,NULL,0)) > 0){
                console.info("Inference Process",ip_pid,"terminated due to timeout");
            }
            ipidTimer.erase(p->ip_id);
            ip_ids[p->ip_id]=0;
            if (ip_list.size()>1)
                ip_list.erase(p);
            else
                ip_list.clear();
            p=find_if(model_map[model].begin(),model_map[model].end(),
                            [i](const InferenceProcess &ip) {return ip.ip_id==i;});
            if (model_map[model].size()>1)
                model_map[model].erase(p);
            else
                model_map.erase(model);
        }
        sleep(1);
    }
}


pid_t IPManager::spawnProcess(char** arg_list, char** env)
{
    pid_t ch_pid = fork();
    if (ch_pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (ch_pid > 0) {
        // cout << "spawn inference process with pid - " << ch_pid << endl;
        return ch_pid;
    } else {
        if (InferType==0)
            execve("infer", arg_list, env);
        else if (InferType==1)
            execve("infer_trt", arg_list, env);
        perror("execve");
        exit(EXIT_FAILURE);
    }
}

// Create inference process for model, with p% GPU, using NVIDIA-MPS
void IPManager::createInferenceProcess(std::string model_name, int ip_id,int SLO){
    int pGPU=getOptGPUpercentage(model_name,SLO);
    int GPUid=chooseGPU(pGPU);
    char mps_setting[50]="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=";
    strcat(mps_setting,std::to_string(pGPU).c_str());
    char *env[50]={mps_setting};
    char ipid_str[2];
    strcpy(ipid_str,std::to_string(ip_id).c_str());
    char _model_name[20];
    strcpy(_model_name,model_name.c_str());
    char _gpuid[2];
    strcpy(_gpuid,std::to_string(GPUid).c_str());
    char infer_type[10];
    if (InferType==0)
        strcpy(infer_type,"infer");
    else if (InferType==1)
        strcpy(infer_type,"infer_trt");
    char* args_list[]={infer_type,_model_name,ipid_str,_gpuid,NULL};
    console.info("Create Inference Process ",ip_id," on GPU",GPUid,"with",pGPU,"%");
    struct InferenceProcess new_ip; 
    new_ip.ip_id=ip_id;
    new_ip.pid = spawnProcess(args_list, env);
    new_ip.IFtime=profiler[model_name][pGPU];
    new_ip.model=model_name;
    new_ip.allocatedGPU=GPUid;
    new_ip.pGPU=pGPU;
    ip_list.push_back(new_ip);
    model_map[model_name].push_back(new_ip);
}

// do inference with InferenceProcess ip_id
int IPManager::infer(void* mem_addr,std::string base64_image, int ip_id){
    ipidTimer[ip_id]=std::chrono::steady_clock::now();       // update timer for this IP
    void* ip_addr=mem_addr+MEM_BLOCK*ip_id;     //determine shared memory address belong to IP
    // Decode image, preprocess and write to shared memory
    std::string decoded_image = base64_decode(base64_image);
    std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
    cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);
    image = preprocess(image, image_height, image_width, mean, std);
    // inference
    ip_locks[ip_id].lock();
    memcpy(ip_addr+1,image.data,image.total() * image.elemSize());  
    memset(ip_addr,1,1);                        // set marker byte=1  
    char* running=static_cast<char*>(ip_addr); 
    while (*running);                           // if marker byte ==0, then inference is finished.
    char *_v1=static_cast<char*>(ip_addr)+1;    // get 2 bytes of result
    char *_v2=static_cast<char*>(ip_addr)+2;
    int v1=int(*_v1);
    if (v1<0)
        v1=256+v1;
    int v2=int(*_v2);
    if (v2<0)
        v2=256+v2;
    ip_locks[ip_id].unlock();
    return v2*256+v1;
}

// Search for best fit Inference Process for the requirement
int IPManager::searchIP(std::string model_name,int slo){
    auto ips=model_map.find(model_name);
    int ip_id=-1;
    if (ips==model_map.end()){
        search_lock.lock();
        ip_id=getAvailableIPID();
        if (ip_id==-1){
            return -1;
        }
        ip_ids[ip_id]=1;
        createInferenceProcess(model_name,ip_id,slo);
        search_lock.unlock();
        return ip_id;
    }
    int mIF=0;
    for (auto& it: ips->second){
        if (it.IFtime<=slo && it.IFtime>mIF){
            mIF=it.IFtime;
            ip_id=it.ip_id;
        }
    }
    return ip_id;
};

int IPManager::getAvailableIPID(){
    for (int i=0;i<NIP_MAX;i++){
        if (ip_ids[i]==0)
            return i;
    }
    return -1;
}

int IPManager::handle(void* sharedMemAddr, std::string image,std::string model_name,int SLO){
    if (!running){
        console.error("IPManger is not running");
        return -1;
    }
    int ip_id= searchIP(model_name,SLO);
    if (ip_id==-1){
        console.info("System overloaded");
        return -1;
    }
    else if (ip_id==-2){
        console.info("SLO is too tight, can not generate any satisfy inference process");
        return -1;
    }
    else {
        // Run model inference
        console.info("Inference with IP",ip_id);
        int pred=infer(sharedMemAddr,image,ip_id);
        return pred;
    }
}

int IPManager::getOptGPUpercentage(std::string model_name,int slo){
    auto model_slo=profiler[model_name];
    int minp=100;
    for (auto& it : model_slo){
        if (it.second<=slo && it.first<minp)
            minp=it.first;
    }
    return minp;
};
int IPManager::chooseGPU(int pGPU){
    int chosen=-1;
    int minG=101;
    //choose best fit GPU for the pGPU demand
    for (int i=0;i<n_GPU;i++){
        if (GPUresources[i]>=pGPU && GPUresources[i]<minG){
            chosen=i;
            minG=GPUresources[i];
        }
    }
    // Greedy mode
    if (chosen==-1){
        console.info("No fit GPU found, just insert IP to the most free GPU");
        chosen=std::distance(GPUresources.begin(),std::max_element(GPUresources.begin(),GPUresources.end()));
    }
    GPUresources[chosen]=GPUresources[chosen]-pGPU;
    return chosen;
};
void IPManager::setInferType(int type){
    InferType=type;
    return;
};

IPManager::~IPManager(){
    // Kill all Inference Processes
    for (int i = 0; i < ip_list.size(); ++i) {
        kill(ip_list[i].pid, SIGTERM);
    }
    std::atomic<int> ip_pid;
    while ((ip_pid = wait(nullptr)) > 0){
        console.info("Inference Process",ip_pid,"terminated");
    }
    shared_memory_object::remove("sharedBuf");
    _exit(EXIT_SUCCESS);
};
