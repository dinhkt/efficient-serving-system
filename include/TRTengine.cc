#include <iostream>
#include <fstream>

#include "TRTengine.h"
#include "NvOnnxParser.h"
#include "consolelog.hpp"

void Logger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

bool TRTengine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

TRTengine::TRTengine(const Options &options)
    : m_options(options) {}

bool TRTengine::build(std::string onnxModelPath) {
    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options);
    console.info("Searching for engine file with name:",m_engineName);

    if (doesFileExist(m_engineName)) {
        console.info("Engine found, not regenerating...");
        return true;
    }

    // Was not able to find the engine file, generate...
    console.info("Engine not found, generating...");

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set the max supported batch size
    builder->setMaxBatchSize(m_options.maxBatchSize);
    // Define an explicit batch size and then create the network.
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Save the input height, width, and channels.
    // Require this info for inference.
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Specify the optimization profiles and the
    IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, inputC, inputH, inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    config->addOptimizationProfile(defaultProfile);

    // Specify all the optimization profiles.
    for (const auto& optBatchSize: m_options.optBatchSizes) {
        if (optBatchSize == 1) {
            continue;
        }

        if (optBatchSize > m_options.maxBatchSize) {
            throw std::runtime_error("optBatchSize cannot be greater than maxBatchSize!");
        }

        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(optBatchSize, inputC, inputH, inputW));
        profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        config->addOptimizationProfile(profile);
    }

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) {
        return false;
    }
    config->setProfileStream(*profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    console.info("Success, saved engine to",m_engineName);

    return true;
}

TRTengine::~TRTengine() {
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
}

bool TRTengine::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    auto cudaRet = cudaStreamCreate(&m_cudaStream);
    if (cudaRet != 0) {
        throw std::runtime_error("Unable to create cuda stream");
    }

    return true;
}

bool TRTengine::runInference(void *image_bytes, int batchSize, std::vector<std::vector<float>>& outputs) {
    auto dims = m_engine->getBindingDimensions(0);
    auto outputL = m_engine->getBindingDimensions(1).d[1];
    Dims4 inputDims = {1, dims.d[1], dims.d[2], dims.d[3]};
    m_context->setBindingDimensions(0, inputDims);
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all input dimensions specified.");
    }

    // Only reallocate buffers if the batch size has changed
    if (m_prevBatchSize != batchSize) {

        m_inputBuff.hostBuffer.resize(inputDims);
        m_inputBuff.deviceBuffer.resize(inputDims);

        Dims2 outputDims {batchSize, outputL};
        m_outputBuff.hostBuffer.resize(outputDims);
        m_outputBuff.deviceBuffer.resize(outputDims);

        m_prevBatchSize = batchSize;
    }

    auto* hostDataBuffer = static_cast<float*>(m_inputBuff.hostBuffer.data());

    // NHWC to NCHW conversion
    for (size_t batch = 0; batch<batchSize;  ++batch) {
        int offset = dims.d[1] * dims.d[2] * dims.d[3] * batch;
        int r = 0 , g = 0, b = 0;
        for (int i = 0; i < dims.d[1] * dims.d[2] * dims.d[3]; ++i) {
            if (i % 3 == 0) {
                hostDataBuffer[offset + r++] = *(reinterpret_cast<float*>(image_bytes) + i);
            } else if (i % 3 == 1) {
                hostDataBuffer[offset + g++ + dims.d[2]*dims.d[3]] = *(reinterpret_cast<float*>(image_bytes) + i);
            } else {
                hostDataBuffer[offset + b++ + dims.d[2]*dims.d[3]*2] = *(reinterpret_cast<float*>(image_bytes) + i);
            }
        }
    }
    // Copy from CPU to GPU
    auto ret = cudaMemcpyAsync(m_inputBuff.deviceBuffer.data(), m_inputBuff.hostBuffer.data(), m_inputBuff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, m_cudaStream);
    if (ret != 0) {
        return false;
    }

    std::vector<void*> predicitonBindings = {m_inputBuff.deviceBuffer.data(), m_outputBuff.deviceBuffer.data()};

    // Run inference.
    bool status = m_context->enqueueV2(predicitonBindings.data(), m_cudaStream, nullptr);
    if (!status) {
        return false;
    }

    // Copy the results back to CPU memory
    ret = cudaMemcpyAsync(m_outputBuff.hostBuffer.data(), m_outputBuff.deviceBuffer.data(), m_outputBuff.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        console.error("Unable to copy buffer from GPU back to CPU");
        return false;
    }

    ret = cudaStreamSynchronize(m_cudaStream);
    if (ret != 0) {
        console.error("Unable to synchronize cuda stream");
        return false;
    }

    // Copy to output
    for (int batch = 0; batch < batchSize; ++batch) {
        std::vector<float> output;
        output.resize(outputL);
        memcpy(output.data(), reinterpret_cast<const char*>(m_outputBuff.hostBuffer.data()) +
        batch * outputL * sizeof(float), outputL * sizeof(float ));
        outputs.emplace_back(std::move(output));
    }

    return true;
}

std::string TRTengine::serializeEngineOptions(const Options &options) {
    std::string engineName = "trt.engine";

    std::vector<std::string> gpuUUIDs;
    getGPUUUIDs(gpuUUIDs);

    if (static_cast<size_t>(options.deviceIndex) >= gpuUUIDs.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    engineName+= "." + gpuUUIDs[options.deviceIndex];
    engineName+= "." +options.model_name;
    // Serialize the specified options into the filename
    if (options.FP16) {
        engineName += ".fp16";
    } else {
        engineName += ".fp32";
    }

    engineName += "." + std::to_string(options.maxBatchSize) + ".";
    for (size_t i = 0; i < m_options.optBatchSizes.size(); ++i) {
        engineName += std::to_string(m_options.optBatchSizes[i]);
        if (i != m_options.optBatchSizes.size() - 1) {
            engineName += "_";
        }
    }

    engineName += "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}

void TRTengine::getGPUUUIDs(std::vector<std::string>& gpuUUIDs) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        char uuid[33];
        for(int b=0; b<16; b++) {
            sprintf(&uuid[b*2], "%02x", (unsigned char)prop.uuid.bytes[b]);
        }

        gpuUUIDs.push_back(std::string(uuid));
    }
}