#ifndef TRTENGINE_H
#define TRTENGINE_H

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "buffers.h"

// Options for the network
struct Options {
    std::string model_name;
    bool FP16 = false;
    std::vector<int32_t> optBatchSizes;
    int32_t maxBatchSize = 16;
    size_t maxWorkspaceSize = 4000000000;
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class TRTengine {
public:
    TRTengine(const Options& options);
    ~TRTengine();
    bool build(std::string onnxModelPath);
    bool loadNetwork();
    bool runInference(void *image_bytes, int batchSize, std::vector<std::vector<float>>& outputs);
private:
    std::string serializeEngineOptions(const Options& options);
    void getGPUUUIDs(std::vector<std::string>& gpuUUIDs);
    bool doesFileExist(const std::string& filepath);

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options& m_options;
    Logger m_logger;
    samplesCommon::ManagedBuffer m_inputBuff;
    samplesCommon::ManagedBuffer m_outputBuff;
    size_t m_prevBatchSize = 0;
    std::string m_engineName;
    cudaStream_t m_cudaStream = nullptr;
};

#endif
