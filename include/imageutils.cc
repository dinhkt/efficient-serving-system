#include "imageutils.h"

// Resize an image to a given size to
cv::Mat __resize_to_a_size(cv::Mat image, int new_height, int new_width) {

  // get original image size
  int org_image_height = image.rows;
  int org_image_width = image.cols;

  // get image area and resized image area
  float img_area = float(org_image_height * org_image_width);
  float new_area = float(new_height * new_width);

  // resize
  cv::Mat image_scaled;
  cv::Size scale(new_width, new_height);

  if (new_area >= img_area) {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_LANCZOS4);
  } else {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_AREA);
  }

  return image_scaled;
}

// Normalize an image by subtracting mean and dividing by standard deviation
cv::Mat __normalize_mean_std(cv::Mat image, std::vector<double> mean, std::vector<double> std) {

  // clone
  cv::Mat image_normalized = image.clone();

  // convert to float
  image_normalized.convertTo(image_normalized, CV_32FC3);

  // subtract mean
  cv::subtract(image_normalized, mean, image_normalized);

  // divide by standard deviation
  std::vector<cv::Mat> img_channels(3);
  cv::split(image_normalized, img_channels);

  img_channels[0] = img_channels[0] / std[0];
  img_channels[1] = img_channels[1] / std[1];
  img_channels[2] = img_channels[2] / std[2];

  cv::merge(img_channels, image_normalized);

  return image_normalized;  
}

// 1. Preprocess
cv::Mat preprocess(cv::Mat image, int new_height, int new_width,
  std::vector<double> mean, std::vector<double> std) {

  // Clone
  cv::Mat image_proc = image.clone();

  // Convert from BGR to RGB
  cv::cvtColor(image_proc, image_proc, cv::COLOR_BGR2RGB);

  // Resize image
  image_proc = __resize_to_a_size(image_proc, new_height, new_width);

  // Convert image to float
  image_proc.convertTo(image_proc, CV_32FC3);

  // 3. Normalize to [0, 1]
  image_proc = image_proc / 255.0;

  // 4. Subtract mean and divide by std
  image_proc = __normalize_mean_std(image_proc, mean, std);

  return image_proc;
}

// Convert a vector of images to torch Tensor
torch::Tensor convert_images_to_tensor(std::vector<cv::Mat> images) {

  int n_images = images.size();
  int n_channels = images[0].channels();
  int height = images[0].rows;
  int width = images[0].cols;

  int image_type = images[0].type();

  // Image Type must be one of CV_8U, CV_32F, CV_64F
  assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

  std::vector<int64_t> dims = {1, height, width, n_channels};
  std::vector<int64_t> permute_dims = {0, 3, 1, 2};

  std::vector<torch::Tensor> images_as_tensors;
  for (int i = 0; i != n_images; i++) {
    cv::Mat image = images[i].clone();

    torch::Tensor image_as_tensor;
    if (image_type % 8 == 0) {
      torch::TensorOptions options(torch::kUInt8);
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
    } else if ((image_type - 5) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat32);
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
    } else if ((image_type - 6) % 8 == 0) {
      torch::TensorOptions options(torch::kFloat64);
      image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
    }

    image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
    image_as_tensor = image_as_tensor.toType(torch::kFloat32);
    images_as_tensors.push_back(image_as_tensor);
  }

  torch::Tensor output_tensor = torch::cat(images_as_tensors, 0);

  return output_tensor;
}

// Softmax
std::vector<float> __softmax(std::vector<float> unnorm_probs) {

  int n_classes = unnorm_probs.size();

  // 1. Partition function
  float log_sum_of_exp_unnorm_probs = 0;
  for (auto& n : unnorm_probs) {
    log_sum_of_exp_unnorm_probs += std::exp(n);
  }
  log_sum_of_exp_unnorm_probs = std::log(log_sum_of_exp_unnorm_probs);

  // 2. normalize
  std::vector<float> probs(n_classes);
  for (int class_idx = 0; class_idx != n_classes; class_idx++) {
    probs[class_idx] = std::exp(unnorm_probs[class_idx] - log_sum_of_exp_unnorm_probs);
  }

  return probs;
}

// Convert output tensor to vector of floats
std::vector<float> get_outputs(torch::Tensor output) {

  int ndim = output.ndimension();
  assert(ndim == 2);

  torch::ArrayRef<int64_t> sizes = output.sizes();
  int n_samples = sizes[0];
  int n_classes = sizes[1];

  assert(n_samples == 1);

  std::vector<float> unnorm_probs(output.data_ptr<float>(),
                                  output.data_ptr<float>() + (n_samples * n_classes));

  // Softmax
  std::vector<float> probs = __softmax(unnorm_probs);

  return probs;
}


//Postprocess
std::tuple<int, float> postprocess(std::vector<float> probs) {
  auto prob = std::max_element(probs.begin(), probs.end());
  auto label_idx = std::distance(probs.begin(), prob);
  float prob_float = *prob;
  return std::make_tuple(label_idx, prob_float);
}
