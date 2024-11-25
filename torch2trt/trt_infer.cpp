#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

struct Detection
{
    float bbox[4]; // x1, y1, x2, y2
    float confidence;
    float landmarks[10]; // 5 landmarks
};

// 函数声明
cv::Mat letterbox(const cv::Mat &img, int inputW, int inputH, cv::Vec4d &params);
void preprocessImage(const cv::Mat &img, float *gpuInput, int inputH, int inputW, cv::Vec4d &params);
void doInference(IExecutionContext &context, float *gpuInput, float *gpuOutput, int batchSize);
void drawBoundingBoxes(cv::Mat &img, const std::vector<Detection> &detections);
void scaleCoords(const cv::Size &img0_shape, const cv::Vec4d &params, Detection &det);
float computeIoU(const Detection &det_a, const Detection &det_b);
std::vector<Detection> nonMaxSuppression(const std::vector<Detection> &detections, float iouThreshold);

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine file> <image file>" << std::endl;
        return -1;
    }

    std::string engineFile = argv[1];
    std::string imageFile = argv[2];

    std::cout << "Loading engine..." << std::endl;
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << engineFile << std::endl;
        return -1;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime *runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return -1;
    }

    ICudaEngine *engine = runtime->deserializeCudaEngine(buffer.data(), size);
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return -1;
    }

    IExecutionContext *context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(imageFile);
    if (img.empty()) {
        std::cerr << "Error loading image: " << imageFile << std::endl;
        return -1;
    }

    float *gpuInput;
    float *gpuOutput;
    cudaError_t status;

    status = cudaMalloc(reinterpret_cast<void **>(&gpuInput), 3 * 640 * 640 * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for gpuInput: " << cudaGetErrorString(status) << std::endl;
        return -1;
    }

    status = cudaMalloc(reinterpret_cast<void **>(&gpuOutput), 25200 * 16 * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for gpuOutput: " << cudaGetErrorString(status) << std::endl;
        cudaFree(gpuInput);
        return -1;
    }

    // 预处理图像
    cv::Vec4d params; // [gain, pad_w, pad_h, 1/gain]
    preprocessImage(img, gpuInput, 640, 640, params);

    // 推理
    std::cout << "Running inference..." << std::endl;
    // 推理开始时间
    auto start = std::chrono::high_resolution_clock::now();
    doInference(*context, gpuInput, gpuOutput, 1);
    // 推理结束时间
    auto end = std::chrono::high_resolution_clock::now();
    // 打印推理时间
    std::cout << "Inference time: " << std::chrono::duration<float, std::milli>(end - start).count() << " ms"
              << std::endl;

    // 复制输出到CPU
    std::vector<float> cpuOutput(25200 * 16);
    status = cudaMemcpy(cpuOutput.data(), gpuOutput, 25200 * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for copying output to host: " << cudaGetErrorString(status) << std::endl;
        cudaFree(gpuInput);
        cudaFree(gpuOutput);
        return -1;
    }

    float confThreshold = 0.8;
    float iouThreshold = 0.5;

    // 解析检测结果
    std::cout << "Post-processing detections..." << std::endl;
    std::vector<Detection> detections;

    for (size_t i = 0; i < cpuOutput.size(); i += 16) {
        float confidence = cpuOutput[i + 4];
        if (confidence < confThreshold)
            continue;

        Detection det;
        float center_x = cpuOutput[i];
        float center_y = cpuOutput[i + 1];
        float width = cpuOutput[i + 2];
        float height = cpuOutput[i + 3];

        det.bbox[0] = center_x - width / 2; // x1
        det.bbox[1] = center_y - height / 2; // y1
        det.bbox[2] = center_x + width / 2; // x2
        det.bbox[3] = center_y + height / 2; // y2

        det.confidence = confidence;

        // 复制关键点坐标
        for (int j = 0; j < 10; ++j) {
            det.landmarks[j] = cpuOutput[i + 5 + j];
        }

        // 缩放坐标到原图尺寸
        scaleCoords(img.size(), params, det);

        detections.push_back(det);
    }

    // 非极大值抑制
    detections = nonMaxSuppression(detections, iouThreshold);

    for (const auto &det : detections) {
        std::cout << "bbox: [" << det.bbox[0] << ", " << det.bbox[1] << ", " << det.bbox[2] << ", " << det.bbox[3]
                  << "], confidence: " << det.confidence << std::endl;
    }

    std::cout << "Drawing bounding boxes..." << std::endl;
    drawBoundingBoxes(img, detections);

    std::cout << "Saving result image..." << std::endl;
    cv::imwrite("result.jpg", img);

    cudaFree(gpuInput);
    cudaFree(gpuOutput);

    return 0;
}

// 函数实现
cv::Mat letterbox(const cv::Mat &img, int inputW, int inputH, cv::Vec4d &params)
{
    int width = img.cols;
    int height = img.rows;
    float r = std::min((float)inputW / width, (float)inputH / height);
    int new_width = r * width;
    int new_height = r * height;
    int pad_w = inputW - new_width;
    int pad_h = inputH - new_height;
    pad_w /= 2;
    pad_h /= 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_width, new_height));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_h, pad_h, pad_w, pad_w, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    // 保存缩放和填充参数以供后续使用
    params[0] = r; // 缩放比例
    params[1] = pad_w; // 横向填充
    params[2] = pad_h; // 纵向填充
    params[3] = 1 / r; // 缩放比例的倒数

    return padded;
}

void preprocessImage(const cv::Mat &img, float *gpuInput, int inputH, int inputW, cv::Vec4d &params)
{
    cv::Mat resized = letterbox(img, inputW, inputH, params);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    cudaMemcpy(gpuInput, channels[0].data, inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInput + inputH * inputW, channels[1].data, inputH * inputW * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInput + 2 * inputH * inputW, channels[2].data, inputH * inputW * sizeof(float),
               cudaMemcpyHostToDevice);
}

void doInference(IExecutionContext &context, float *gpuInput, float *gpuOutput, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();
    void *buffers[2];

    // 获取输入和输出绑定名称
    const char *inputName = engine.getIOTensorName(0);
    const char *outputName = engine.getIOTensorName(1);

    // 设置输入和输出的地址
    context.setTensorAddress(inputName, gpuInput);
    context.setTensorAddress(outputName, gpuOutput);

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 推理，使用 enqueueV3
    context.enqueueV3(stream);

    // 同步 CUDA 流
    cudaStreamSynchronize(stream);

    // 释放流
    cudaStreamDestroy(stream);
}

void scaleCoords(const cv::Size &img0_shape, const cv::Vec4d &params, Detection &det)
{
    float gain = params[0];
    float pad_w = params[1];
    float pad_h = params[2];

    // 坐标缩放并去除填充
    det.bbox[0] = (det.bbox[0] - pad_w) / gain;
    det.bbox[1] = (det.bbox[1] - pad_h) / gain;
    det.bbox[2] = (det.bbox[2] - pad_w) / gain;
    det.bbox[3] = (det.bbox[3] - pad_h) / gain;

    // 缩放关键点坐标
    for (int i = 0; i < 5; ++i) {
        det.landmarks[i * 2] = (det.landmarks[i * 2] - pad_w) / gain;
        det.landmarks[i * 2 + 1] = (det.landmarks[i * 2 + 1] - pad_h) / gain;
    }

    // 限制坐标在图像边界内
    det.bbox[0] = std::clamp(det.bbox[0], 0.0f, (float)img0_shape.width);
    det.bbox[1] = std::clamp(det.bbox[1], 0.0f, (float)img0_shape.height);
    det.bbox[2] = std::clamp(det.bbox[2], 0.0f, (float)img0_shape.width);
    det.bbox[3] = std::clamp(det.bbox[3], 0.0f, (float)img0_shape.height);

    for (int i = 0; i < 5; ++i) {
        det.landmarks[i * 2] = std::clamp(det.landmarks[i * 2], 0.0f, (float)img0_shape.width);
        det.landmarks[i * 2 + 1] = std::clamp(det.landmarks[i * 2 + 1], 0.0f, (float)img0_shape.height);
    }
}

void drawBoundingBoxes(cv::Mat &img, const std::vector<Detection> &detections)
{
    for (const auto &det : detections) {
        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

        // 绘制关键点
        for (int j = 0; j < 5; ++j) {
            int lx = static_cast<int>(det.landmarks[j * 2]);
            int ly = static_cast<int>(det.landmarks[j * 2 + 1]);
            cv::circle(img, cv::Point(lx, ly), 2, cv::Scalar(0, 0, 255), -1);
        }

        std::string label = cv::format("Conf: %.2f", det.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        y1 = std::max(y1, labelSize.height);
        cv::rectangle(img, cv::Point(x1, y1 - labelSize.height), cv::Point(x1 + labelSize.width, y1 + baseLine),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img, label, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

float computeIoU(const Detection &det_a, const Detection &det_b)
{
    float x1 = std::max(det_a.bbox[0], det_b.bbox[0]);
    float y1 = std::max(det_a.bbox[1], det_b.bbox[1]);
    float x2 = std::min(det_a.bbox[2], det_b.bbox[2]);
    float y2 = std::min(det_a.bbox[3], det_b.bbox[3]);

    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = (det_a.bbox[2] - det_a.bbox[0]) * (det_a.bbox[3] - det_a.bbox[1]);
    float areaB = (det_b.bbox[2] - det_b.bbox[0]) * (det_b.bbox[3] - det_b.bbox[1]);

    float unionArea = areaA + areaB - interArea;
    if (unionArea == 0)
        return 0;
    return interArea / unionArea;
}

std::vector<Detection> nonMaxSuppression(const std::vector<Detection> &detections, float iouThreshold)
{
    std::vector<Detection> result;
    std::vector<Detection> dets = detections;

    // 按置信度从高到低排序
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) { return a.confidence > b.confidence; });

    while (!dets.empty()) {
        Detection best = dets[0];
        result.push_back(best);
        dets.erase(dets.begin());

        for (auto it = dets.begin(); it != dets.end();) {
            float iou = computeIoU(best, *it);
            if (iou > iouThreshold) {
                it = dets.erase(it);
            } else {
                ++it;
            }
        }
    }
    return result;
}
