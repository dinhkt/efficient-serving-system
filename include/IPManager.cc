#include "IPManager.h"

using namespace std;
using namespace boost::interprocess;


void IPManager::run(){
    running=true;
    // info of GPUs resources available
    n_GPU=at::cuda::device_count();
    for (int i=0;i<n_GPU;i++){
        GPUresources.push_back(100);
    }
    cout<<"System has "<<n_GPU<<" GPU"<<endl;
    boost::thread th(&IPManager::IPsTimer, this);
    
    string row;
    ifstream  data("../profiled_data/profiler.txt");
    if (data.is_open())
    {
        string tmp,name,p,bs,t;
        int i=0;
        while (getline(data,row)){
            stringstream X(row);
            string arr[4];
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
        cout<<"Profiler file not found"<<endl;
        return;
    }
    
}
void IPManager::IPsTimer(){
    while (true){
        auto now=chrono::steady_clock::now();
        vector<int> expired_list;
        for(auto iter = ipidTimer.begin(); iter != ipidTimer.end(); ++iter){
            if (chrono::duration_cast<chrono::seconds>(now-iter->second).count()>time_out)
                expired_list.push_back(iter->first);
        }

        for (auto i: expired_list){
            auto p=find_if(ip_list.begin(),ip_list.end(),
                            [i](const InferenceProcess& ip) {return ip.ip_id==i;});
            string model=p->model;
            kill(p->pid,SIGTERM);
            GPUresources[p->allocatedGPU]+=p->pGPU;
            atomic<int> ip_pid;
            while ((ip_pid = waitpid(p->pid,NULL,0)) > 0){
                cout << "Inference Process " << ip_pid << " terminated due to timeout" << endl;
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
IPManager::~IPManager(){
    // Kill all Inference Processes
    for (int i = 0; i < ip_list.size(); ++i) {
        kill(ip_list[i].pid, SIGTERM);
    }
    atomic<int> ip_pid;
    while ((ip_pid = wait(nullptr)) > 0);
    shared_memory_object::remove("sharedBuf");
    _exit(EXIT_SUCCESS);
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
        execve("infer", arg_list, env);
        perror("execve");
        exit(EXIT_FAILURE);
    }
}

// Create inference process for model, with p% GPU, using NVIDIA-MPS
void IPManager::createInferenceProcess(string model_name, int ip_id,int SLO){
    int pGPU=getOptGPUpercentage(model_name,SLO);
    int GPUid=chooseGPU(pGPU);
    char mps_setting[50]="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=";
    strcat(mps_setting,to_string(pGPU).c_str());
    char *env[50]={mps_setting};
    char ipid_str[2];
    strcpy(ipid_str,to_string(ip_id).c_str());
    char _model_name[20];
    strcpy(_model_name,model_name.c_str());
    char _gpuid[2];
    strcpy(_gpuid,to_string(GPUid).c_str());
    char* args_list[]={"infer",_model_name,ipid_str,_gpuid,NULL};
    cout<<"Create Inference Process "<<ip_id<<" on GPU"<<GPUid<< " with "<<pGPU<<"%"<<endl;
    struct InferenceProcess new_ip; 
    new_ip.ip_id=ip_id;
    new_ip.pid = spawnProcess(args_list, env);
    new_ip.IFtime=profiler[model_name][pGPU];
    new_ip.model=model_name;
    new_ip.allocatedGPU=GPUid;
    new_ip.pGPU=pGPU;
    ip_list.push_back(new_ip);
    model_map[model_name].push_back(new_ip);
    ip_ids[ip_id]=1;
}

// do inference with InferenceProcess ip_id
int IPManager::infer(void* mem_addr,string base64_image, int ip_id){
    ipidTimer[ip_id]=chrono::steady_clock::now();       // update timer for this IP
    void* ip_addr=mem_addr+MEM_BLOCK*ip_id;     //determine shared memory address belong to IP
    ip_locks[ip_id].lock();
    memcpy(ip_addr,base64_image.c_str(),base64_image.length());    
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
int IPManager::searchIP(string model_name,int slo){
    search_lock.lock();
    auto ips=model_map.find(model_name);
    int ip_id=-1;
    if (ips==model_map.end()){
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
    search_lock.unlock();
    return ip_id;
};

int IPManager::getAvailableIPID(){
    for (int i=0;i<NIP_MAX;i++){
        if (ip_ids[i]==0)
            return i;
    }
    return -1;
}

int IPManager::handle(void* sharedMemAddr, string image,string model_name,int SLO){
    if (!running){
        cout<<"IPManger is not running"<<endl;
        return -1;
    }
    int ip_id= searchIP(model_name,SLO);
    // If not found, create Inference Process
    if (ip_id==-1){
        ip_id=getAvailableIPID();
        if (ip_id==-1){
            return -1;
        }
        createInferenceProcess(model_name,ip_id,SLO);
    }
    while (ip_id!=-1){
        // Run model inference
        cout<<"Inference with IP "<<ip_id<<endl;
        int pred=infer(sharedMemAddr,image,ip_id);
        return pred;
    }
}

int IPManager::getOptGPUpercentage(string model_name,int slo){
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
    if (chosen==-1){
        cout<<"No GPU found, just insert IP to the most free GPU"<<endl;
        chosen=distance(GPUresources.begin(),max_element(GPUresources.begin(),GPUresources.end()));
    }
    int available_resources=GPUresources[chosen]-pGPU;
    GPUresources[chosen]=(available_resources>0) ? available_resources:0;
    return chosen;
};
