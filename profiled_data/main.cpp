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

pid_t spawnProcess(char** arg_list, char** env)
{
    pid_t ch_pid = fork();
    if (ch_pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (ch_pid > 0) {
        // cout << "spawn inference process with pid - " << ch_pid << endl;
        std::cout<<"Spawn process"<<std::endl;
        return ch_pid;
    } else {
        execve("trt_profiler", arg_list, env);
        perror("execve");
        exit(EXIT_FAILURE);
    }
}

// Create inference process for model, with p% GPU, using NVIDIA-MPS
void createInferenceProcess(std::string model_name, int pGPU){
    char mps_setting[50]="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=";
    strcat(mps_setting,std::to_string(pGPU).c_str());
    char *env[50]={mps_setting};
    char _pGPU[10];
    strcpy(_pGPU,std::to_string(pGPU).c_str());
    char _model_name[20];
    strcpy(_model_name,model_name.c_str());
    char* args_list[]={"trt_profiler",_model_name,_pGPU,NULL};
    spawnProcess(args_list, env);
}

int main(){
    std::vector<std::string> models={"resnet18","resnet50","vgg16","vgg19"};
    std::atomic<int> ip_pid;
    for (auto model:models){
        for (int p=10;p<110;p+=10){
            createInferenceProcess(model,p);
            while ((ip_pid = wait(nullptr)) > 0){
                std::cout<<"Inference Process"<<ip_pid<<"terminated";
            }
        }
    }
}