#ifndef IMAGEUTILS_H 
#define IMAGEUTILS_H

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>
#include <vector>
#include <math.h>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat preprocess(cv::Mat, int, int,
  std::vector<double>,
  std::vector<double>);
torch::Tensor convert_images_to_tensor(std::vector<cv::Mat> images);
std::vector<float> get_outputs(torch::Tensor output);

#endif
