#include "IPManager.h"
#include "consolelog.hpp"

using namespace boost::interprocess;


void IPManager::run(){
    running=true;
    // info of GPUs resources available
    n_GPU=at::cuda::device_count();
    for (int i=0;i<n_GPU;i++){
        GPUresources.emplace_back(100);
    }
    console.info("System has",n_GPU,"GPU");
    boost::thread _SC_THREAD_ATTR_STACKADDR(&IPManager::IPsTimer, this);
    
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
                    profiler[arr[0]+"_"+arr[2]][stoi(arr[1])]=stof(arr[3]);
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

        // for (auto i: expired_list){
        //     auto p=find_if(ip_list.begin(),ip_list.end(),
        //                     [i](const InferenceProcess& ip) {return ip.ip_id==i;});
        //     std::string model=p->model;
        //     kill(p->pid,SIGTERM);
        //     GPUresources[p->allocatedGPU]+=p->pGPU;
        //     std::atomic<int> ip_pid;
        //     while ((ip_pid = waitpid(p->pid,NULL,0)) > 0){
        //         console.info("Inference Process",ip_pid,"terminated due to timeout");
        //     }
        //     ipidTimer.erase(p->ip_id);
        //     ip_ids[p->ip_id]=0;
        //     if (ip_list.size()>1)
        //         ip_list.erase(p);
        //     else
        //         ip_list.clear();
        // }
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

// Create inference process for model, with p% GPU on GPUid, using NVIDIA-MPS
int IPManager::createInferenceProcess(std::string model_name,int ip_id, int slo, int batch_size,int service_type){
    int pGPU=getOptGPUpercentage(model_name,slo,batch_size);
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
    char infer_type[20];
    if (InferType==TORCH_CPP)
        strcpy(infer_type,"infer");
    else if (InferType==TENSOR_RT)
        strcpy(infer_type,"infer_trt");
    char* args_list[]={infer_type,_model_name,ipid_str,_gpuid,NULL};
    console.info("Create Inference Process ",ip_id," running on GPU", GPUid,"with",pGPU,"%");
    InferenceProcess* new_ip=new InferenceProcess; 
    new_ip->ip_id=ip_id;
    new_ip->pid = spawnProcess(args_list, env);
    new_ip->rid = 0;
    new_ip->IFtime=profiler[model_name+"_"+std::to_string(batch_size)][pGPU];
    new_ip->model=model_name;
    new_ip->allocatedGPU=GPUid;
    new_ip->pGPU=pGPU;
    new_ip->service_type=service_type;
    // Launch worker
    ip_list.push_back(new_ip);
    model_map[model_name+"_"+std::to_string(batch_size)].push_back(new_ip);
    boost::thread thr(&IPManager::infer, this, ip_id);
    ip_ids[ip_id]=1;
    return ip_id;
}

// inference worker of InferenceProcess ip_id
void IPManager::infer(int ip_id){
    void* ip_addr=baseMem+MEM_BLOCK*ip_id;     //determine shared memory address belong to IP
    sleep(2);
    while (true){
        while (ip_ids[ip_id] && !inference_queues[ip_id].empty()){
            auto ir=inference_queues[ip_id].front(); // get front inference request from queue
            memcpy(ip_addr+1,ir->data,ir->data_size);
            memset(ip_addr,ir->batch_size,1);                        // set marker byte is first byte, first byte store the batch size
            char* running=static_cast<char*>(ip_addr); 
            while (*running);                           // if marker byte == 0, then inference is finished.
            if (ip_list[ip_id]->service_type==IMAGE_CLASSIFICATION){
                ip_list[ip_id]->outputs[ir->rid]=IC_postprocess(ip_addr,ir->batch_size);
                ip_list[ip_id]->completed[ir->rid]=1;
            }
            free(ir->data);
            delete ir;
            inference_queues[ip_id].pop();
        }
        if (!ip_ids[ip_id])
        std::terminate();
    }
}

// Search for best fit Inference Process for the requirement
int IPManager::searchIP(std::string model_name,int slo,int batch_size){
    console.info("key:",model_name+"_"+std::to_string(batch_size));
    auto ips=model_map.find(model_name+"_"+std::to_string(batch_size));
    int ip_id=-1;
    if (ips!=model_map.end()){
        int mIF=0;
        for (auto& it: ips->second){
            if (it->IFtime<=slo && it->IFtime>mIF){
                mIF=it->IFtime;
                ip_id=it->ip_id;
            }
        }
    };
    return ip_id;
};

int IPManager::getAvailableIPID(){
    for (int i=0;i<NIP_MAX;i++){
        if (ip_ids[i]==0)
            return i;
    }
    return -1;
}


std::vector<int> IPManager::handle(std::vector<std::string> images,std::string model_name,int SLO, int batch_size, int service_type){
    console.info("handle");
    if (!running){
        console.error("IPManger is not running");
        return std::vector<int>();
    }
    search_lock.lock();
    int ip_id= searchIP(model_name,SLO,batch_size);
    if (ip_id==-1){
        ip_id=getAvailableIPID();
        if (ip_id==-1){
            console.error("System overloaded");
            return std::vector<int>();
        }
        createInferenceProcess(model_name,ip_id,SLO,batch_size,service_type);
    }
    search_lock.unlock();
    // Run model inference
    console.info("Inference with IP",ip_id);
    ipidTimer[ip_id]=std::chrono::steady_clock::now();       // update timer for this IP
    inference_request* ir=new inference_request;
    int this_rid=ip_list[ip_id]->gen_rid();
    ir->rid=this_rid;
    if (ip_list[ip_id]->service_type==IMAGE_CLASSIFICATION){
        ir->data_size=IC_preprocess(&ir->data,images);
        ir->batch_size=batch_size;
    }
    else
        console.error("Service type not supported");
    // Push to inference queue and wait for result 
    inference_queues[ip_id].push(ir); 
    while (!ip_list[ip_id]->completed[this_rid]);
    std::vector<int> preds=ip_list[ip_id]->outputs[this_rid];
    ip_list[ip_id]->completed[this_rid]=0;
    return preds;
}

int IPManager::getOptGPUpercentage(std::string model_name,int slo,int batch_size){
    auto model_slo=profiler[model_name+"_"+std::to_string(batch_size)];
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

void IPManager::set_baseMem(void* mem_addr){
    baseMem=mem_addr;
    return;
}
IPManager::~IPManager(){
    // Kill all Inference Processes
    for (int i = 0; i < ip_list.size(); ++i) {
        kill(ip_list[i]->pid, SIGTERM);
    }
    std::atomic<int> ip_pid;
    while ((ip_pid = wait(nullptr)) > 0){
        console.info("Inference Process",ip_pid,"terminated");
    }
    shared_memory_object::remove("sharedBuf");
    _exit(EXIT_SUCCESS);
};
