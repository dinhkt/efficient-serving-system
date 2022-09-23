#ifndef IPMANAGER_H
#define IPMANAGER_H

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
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

#define MEM_BLOCK 1000000        // Memory block size for each inference process
#define PORT 8081       
#define NIP_MAX 100             // Maximum number of active inference processes
#define TIMEOUT 30
typedef int pid_t;

using namespace std;
struct InferenceProcess{
    int ip_id;
    int pid;
    string model;
    int allocatedGPU;
    int pGPU;
    int IFtime;
};

class IPManager{
private:
    boost::mutex ip_locks[NIP_MAX];
    atomic<int> ip_ids[NIP_MAX]={};
    boost::mutex search_lock;
    int time_out=TIMEOUT;
    unordered_map<int,chrono::steady_clock::time_point> ipidTimer;

    pid_t spawnProcess(char** arg_list, char** env);
    void IPsTimer();
    int getAvailableIPID();
    int searchIP(string model_name,int slo);
    int getOptGPUpercentage(string model_name,int slo);
    int chooseGPU(int pGPU);
public: 
    vector<InferenceProcess> ip_list;
    unordered_map<string,vector<InferenceProcess>> model_map; 
    unordered_map<std::string,unordered_map<int,float>> profiler;
    bool running=false;

    void run();
    void createInferenceProcess(string model_name, int ip_id,int SLO);
    int infer(void* mem_addr,string base64_image, int ip_id);
    int handle(void* sharedMemAddr, string image,string model_name,int SLO);
    ~IPManager();
};

#endif