#ifndef IPMANAGER_H
#define IPMANAGER_H

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <torch/cuda.h>
#include <ATen/cuda/CUDAEvent.h>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <filesystem>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h> 
#include <errno.h>
#include <vector>
#include <boost/thread.hpp>
#include <unordered_map>
#include <chrono>
#include<fstream>
#include <sstream>
#include "imageutils.h"
#include "base64.h"
#include <queue>

#define MEM_BLOCK 100000000        // Memory block size for each inference process      
#define NIP_MAX 20             // Maximum number of active inference processes
#define TIMEOUT 120
typedef int pid_t;

enum inference_backends{TORCH_CPP,TENSOR_RT};

enum services{
    IMAGE_CLASSIFICATION
};

struct inference_request{
    int rid;
    void *data;
    size_t data_size;
    int batch_size;
    int type;
};

struct InferenceProcess{
    int ip_id;
    int pid;
    std::string model;
    int allocatedGPU;
    int pGPU;
    int IFtime;
    int service_type;
    std::atomic<int> rid;
    std::atomic<int> completed[100] = {};
    std::vector<int> outputs[100];
    boost::mutex id_lock;
    int gen_rid()
    {
        id_lock.lock();
        int cur_rid=rid;
        rid+=1;
        if (rid==100)
            rid=0;
        id_lock.unlock();
        return cur_rid;
    }
};

class IPManager{
private:
    boost::mutex ip_locks[NIP_MAX];
    std::atomic<int> ip_ids[NIP_MAX]={};
    boost::mutex search_lock;
    int time_out=TIMEOUT;
    int InferType=0;
    std::unordered_map<int,std::chrono::steady_clock::time_point> ipidTimer;
    std::unordered_map<std::string,std::vector<InferenceProcess*>> model_map; 
    std::unordered_map<std::string,std::unordered_map<int,float>> profiler;
    std::queue<inference_request*> inference_queues[NIP_MAX];
    std::vector<int> GPUresources;
    void *baseMem;
    
    int createInferenceProcess(std::string model_name,int ip_id, int slo, int batch_size,int service_type);
    void infer(int ip_id);
    pid_t spawnProcess(char** arg_list, char** env);
    void IPsTimer();
    int getAvailableIPID();
    int searchIP(std::string model_name,int slo,int batch_size);
    int getOptGPUpercentage(std::string model_name,int slo,int batch_size);
    int chooseGPU(int pGPU);
public: 
    std::vector<InferenceProcess*> ip_list;
    bool running=false;
    int n_GPU=0;

    void run();
    std::vector<int> handle(std::vector<std::string> images,std::string model_name,int SLO, int batch_size, int service_type);
    void setInferType(int type);
    void set_baseMem(void *mem_addr);
    ~IPManager();
};

#endif