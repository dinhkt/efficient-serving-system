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

#define MEM_BLOCK 1000000        // Memory block size for each inference process
#define PORT 8081       
#define NIP_MAX 100             // Maximum number of active inference processes
#define TIMEOUT 100
typedef int pid_t;

struct InferenceProcess{
    int ip_id;
    int pid;
    std::string model;
    int allocatedGPU;
    int pGPU;
    int IFtime;
};

class IPManager{
private:
    boost::mutex ip_locks[NIP_MAX];
    std::atomic<int> ip_ids[NIP_MAX]={};
    boost::mutex search_lock;
    int time_out=TIMEOUT;
    int InferType=0;
    std::unordered_map<int,std::chrono::steady_clock::time_point> ipidTimer;
    std::unordered_map<std::string,std::vector<InferenceProcess>> model_map; 
    std::unordered_map<std::string,std::unordered_map<int,float>> profiler;
    std::vector<int> GPUresources;
    // Preprocess params
    int image_height = 224;
    int image_width = 224; 
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};

    void createInferenceProcess(std::string model_name, int ip_id,int SLO);
    int infer(void* mem_addr,std::string base64_image, int ip_id);
    pid_t spawnProcess(char** arg_list, char** env);
    void IPsTimer();
    int getAvailableIPID();
    int searchIP(std::string model_name,int slo);
    int getOptGPUpercentage(std::string model_name,int slo);
    int chooseGPU(int pGPU);
public: 
    std::vector<InferenceProcess> ip_list;
    bool running=false;
    int n_GPU=0;

    void run();
    int handle(void* sharedMemAddr, std::string image,std::string model_name,int SLO);
    void setInferType(int type);
    ~IPManager();
};

#endif