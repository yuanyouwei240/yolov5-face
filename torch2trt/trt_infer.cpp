#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace nvinfer1;

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

cv::Mat letterbox(const cv::Mat& img, int inputW, int inputH, cv::Vec4d& params) {
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
    cv::copyMakeBorder(resized, padded, pad_h, pad_h, pad_w, pad_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 保存缩放和填充参数以供后续使用
    params[0] = r;         // 缩放比例
    params[1] = pad_w;     // 横向填充
    params[2] = pad_h;     // 纵向填充
    params[3] = 1 / r;     // 缩放比例的倒数

    return padded;
}

void preprocessImage(const cv::Mat& img, float* gpuInput, int inputH, int inputW, cv::Vec4d& params) {
    cv::Mat resized = letterbox(img, inputW, inputH, params);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    cudaMemcpy(gpuInput, channels[0].data, inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInput + inputH * inputW, channels[1].data, inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuInput + 2 * inputH * inputW, channels[2].data, inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);
}

void doInference(IExecutionContext& context, float* gpuInput, float* gpuOutput, int batchSize) {
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex = engine.getBindingIndex("output");
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * 640 * 640 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * 25200 * 16 * sizeof(float));
    cudaMemcpy(buffers[inputIndex], gpuInput, batchSize * 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice);
    context.enqueueV2(buffers, 0, nullptr);
    cudaMemcpy(gpuOutput, buffers[outputIndex], batchSize * 25200 * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

struct Detection {
    float bbox[4];  // x1, y1, x2, y2
    float confidence;
    float landmarks[10];  // 5 landmarks
};

float computeIoU(const float* bbox1, const float* bbox2) {
    float x1 = std::max(bbox1[0], bbox2[0]);
    float y1 = std::max(bbox1[1], bbox2[1]);
    float x2 = std::min(bbox1[2], bbox2[2]);
    float y2 = std::min(bbox1[3], bbox2[3]);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    float unionArea = area1 + area2 - intersection;

    return intersection / unionArea;
}

std::vector<Detection> nonMaxSuppression(const std::vector<Detection>& detections, float iouThreshold) {
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        const Detection& det = detections[i];
        result.push_back(det);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            const Detection& det2 = detections[j];
            float iou = computeIoU(det.bbox, det2.bbox);
            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

void drawBoundingBoxes(cv::Mat& img, const std::vector<Detection>& detections, float confThreshold) {
    for (const auto& det : detections) {
        if (det.confidence > confThreshold) {
            int x1 = static_cast<int>(det.bbox[0]);
            int y1 = static_cast<int>(det.bbox[1]);
            int x2 = static_cast<int>(det.bbox[2]);
            int y2 = static_cast<int>(det.bbox[3]);
            std::cout << "x1: " << x1 << ", y1: " << y1 << ", x2: " << x2 << ", y2: " << y2 << std::endl;
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

            // Draw landmarks
            for (int j = 0; j < 5; ++j) {
                int lx = static_cast<int>(det.landmarks[j * 2]);
                int ly = static_cast<int>(det.landmarks[j * 2 + 1]);
                cv::circle(img, cv::Point(lx, ly), 2, cv::Scalar(0, 0, 255), -1);
            }

            std::string label = cv::format("Conf: %.2f", det.confidence);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            y1 = std::max(y1, labelSize.height);
            cv::rectangle(img, cv::Point(x1, y1 - labelSize.height),
                          cv::Point(x1 + labelSize.width, y1 + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(img, label, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
}

void scaleCoords(const cv::Size& img0_shape, const cv::Vec4d& params, Detection& det) {
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

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine file> <image file>" << std::endl;
        return -1;
    }

    std::string engineFile = argv[1];
    std::string imageFile = argv[2];

    // Load engine
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

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return -1;
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        runtime->destroy();
        return -1;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // Load image
    std::cout << "Loading image..." << std::endl;
    cv::Mat img = cv::imread(imageFile);
    if (img.empty()) {
        std::cerr << "Error loading image: " << imageFile << std::endl;
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // 检查图片尺寸
    cv::Mat resizedImg;
    if (img.cols != 640 || img.rows != 640) {
        std::cerr << "Image size is not 640x640, resizing..." << std::endl;
        cv::resize(img, resizedImg, cv::Size(640, 640));
    } else {
        resizedImg = img;
    }

    // Allocate memory for input and output
    float* gpuInput;
    float* gpuOutput;
    cudaError_t status;

    std::cout << "Allocating GPU memory for input..." << std::endl;
    status = cudaMalloc(reinterpret_cast<void**>(&gpuInput), 3 * 640 * 640 * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for gpuInput: " << cudaGetErrorString(status) << std::endl;
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    std::cout << "Allocating GPU memory for output..." << std::endl;
    status = cudaMalloc(reinterpret_cast<void**>(&gpuOutput), 25200 * 16 * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for gpuOutput: " << cudaGetErrorString(status) << std::endl;
        cudaFree(gpuInput);
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // 预处理图像，获取缩放和填充参数
    cv::Vec4d params;  // params: [gain, pad_w, pad_h, 1/gain]
    std::cout << "Preprocessing image..." << std::endl;
    preprocessImage(img, gpuInput, 640, 640, params);

    // Run inference
    std::cout << "Running inference..." << std::endl;
    doInference(*context, gpuInput, gpuOutput, 1);

    // 检查gpuOutput是否为空
    if (gpuOutput == nullptr) {
        std::cerr << "gpuOutput is null" << std::endl;
        cudaFree(gpuInput);
        cudaFree(gpuOutput);
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }

    // 将gpuOutput的数据复制到CPU
    std::vector<float> cpuOutput(25200 * 16);
    status = cudaMemcpy(cpuOutput.data(), gpuOutput, 25200 * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for copying output to host: " << cudaGetErrorString(status) << std::endl;
        cudaFree(gpuInput);
        cudaFree(gpuOutput);
        context->destroy();
        engine->destroy();
        runtime->destroy();
        return -1;
    }
    float confThreshold = 0.8;

    // 在将预测结果转换为 Detection 结构并缩放到原图尺寸时
    std::vector<Detection> detections;
    // 在处理模型输出时
    for (size_t i = 0; i < cpuOutput.size(); i += 16)
    {
        Detection det;

        // 确认模型输出的边界框坐标顺序
        float center_x = cpuOutput[i];
        float center_y = cpuOutput[i + 1];
        float width = cpuOutput[i + 2];
        float height = cpuOutput[i + 3];

        // 将中心坐标转换为左上角和右下角坐标
        det.bbox[0] = center_x - width / 2;  // x1
        det.bbox[1] = center_y - height / 2; // y1
        det.bbox[2] = center_x + width / 2;  // x2
        det.bbox[3] = center_y + height / 2; // y2

        det.confidence = cpuOutput[i + 4];

        // 复制特征点（关键点）坐标
        std::copy(cpuOutput.begin() + i + 5, cpuOutput.begin() + i + 15, det.landmarks);

        // 过滤低置信度的检测
        if (det.confidence < confThreshold)
            continue;

        // 缩放坐标到原图尺寸
        scaleCoords(img.size(), params, det);

        detections.push_back(det);
    }
    // // 过滤置信度较低的检测结果
    // detections.erase(std::remove_if(detections.begin(), detections.end(),
    //                                 [confThreshold](const Detection& det) { return det.confidence < confThreshold; }),
    //                  detections.end());

    // // Apply NMS
    // std::cout << "Applying NMS..." << std::endl;
    // detections = nonMaxSuppression(detections, 0.5);

    // 打印抑制后的检测结果
    std::cout << "NMS results:" << std::endl;
    for (const auto& det : detections) {
        std::cout << "BBox: [" << det.bbox[0] << ", " << det.bbox[1] << ", " << det.bbox[2] << ", " << det.bbox[3] << "], "
                  << "Confidence: " << det.confidence << ", "
                  << "Landmarks: [";
        for (int j = 0; j < 10; ++j) {
            std::cout << det.landmarks[j];
            if (j < 9) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Draw bounding boxes on the image
    std::cout << "Drawing bounding boxes..." << std::endl;
    drawBoundingBoxes(img, detections, confThreshold);

    // Save the result image
    std::cout << "Saving result image..." << std::endl;
    cv::imwrite("result.jpg", img);

    // Clean up
    std::cout << "Cleaning up..." << std::endl;
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}