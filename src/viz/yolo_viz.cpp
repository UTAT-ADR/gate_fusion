#include "gate_fusion/yolo_viz.hpp"



using namespace gate;

YOLOSubscriber::YOLOSubscriber(const ros::NodeHandle& nh,
                 const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {

  if (pnh_.getParam("label_file_path", label_file_path) != true) {
    ROS_ERROR("Fail to get label_file_path!");
    exit(1);
  }

  if (pnh_.getParam("img_topic", img_topic) != true) {
    ROS_ERROR("Fail to get img_topic!");
    exit(1);
  }

  if (pnh_.getParam("engine_file_path", engine_file_path) != true) {
    ROS_ERROR("Fail to get engine_file_path!");
    exit(1);
  }

  image_transport::ImageTransport it(nh_);

  imageSub_ = it.subscribe(img_topic, 1, &YOLOSubscriber::imageCallback, this);

  yoloPub_ = it.advertise("yolo_viz", 1000);

  model = new YOLO::DeployDet(engine_file_path);
  labels = YOLOSubscriber::generateLabelColorPairs(label_file_path);

}

void YOLOSubscriber::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  double t1 = ros::Time::now().toSec();
  cv_bridge::CvImageConstPtr BridgePtr;
  BridgePtr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  cv::Mat image;
  cv::cvtColor(BridgePtr->image,image, cv::COLOR_GRAY2RGB);

  YOLO::Image image_in(image.data, image.cols, image.rows);
  double t2 = ros::Time::now().toSec();
  auto result = model->predict(image_in);
  double t3 = ros::Time::now().toSec();
  YOLOSubscriber::visualize(image, result, labels);

  sensor_msgs::ImagePtr yolo_msg = cv_bridge::CvImage(msg->header, "rgb8", image).toImageMsg();
  yoloPub_.publish(yolo_msg);

  double t4 = ros::Time::now().toSec();

  double t_total = (t4 - t1) * 1000;
  double t_pre = (t2 - t1) * 1000;
  double t_infer = (t3 - t2) * 1000;
  double t_post = (t4 - t3) * 1000;

  ROS_INFO("Total: %.2f ms; Preprocess: %.2f ms; Inference: %.2f ms; Visualize: %.2f ms", t_total, t_pre, t_infer, t_post);

}

// Generate label and color pairs
std::vector<std::pair<std::string, cv::Scalar>> YOLOSubscriber::generateLabelColorPairs(const std::string& labelFile) {
    std::ifstream                                   file(labelFile);
    std::vector<std::pair<std::string, cv::Scalar>> labelColorPairs;
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + labelFile);
    }

    auto generateRandomColor = []() {
        std::random_device                 rd;
        std::mt19937                       gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        return cv::Scalar(dis(gen), dis(gen), dis(gen));
    };

    std::string label;
    while (std::getline(file, label)) {
        labelColorPairs.emplace_back(label, generateRandomColor());
    }
    return labelColorPairs;
}

// Visualize detection results
void YOLOSubscriber::visualize(cv::Mat& image, const YOLO::DetectionResult& result, const std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs) {
    for (size_t i = 0; i < result.num; ++i) {
        const auto& box       = result.boxes[i];
        int         cls       = result.classes[i];
        float       score     = result.scores[i];
        const auto& kps       = result.kps[i];
        const auto& label     = labelColorPairs[cls].first;
        const auto& color     = labelColorPairs[cls].second;
        std::string labelText = label + " " + cv::format("%.2f", score);

        // Draw rectangle and label
        int      baseLine;
        cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(image, cv::Point(box.left - (box.right / 2), box.top - (box.bottom / 2)), cv::Point(box.left + (box.right / 2), box.top + (box.bottom / 2)), color, 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left - (box.right / 2), box.top - (box.bottom / 2) - labelSize.height), cv::Point(box.left - (box.right / 2) + labelSize.width, box.top - (box.bottom / 2)), color, -1);
        cv::putText(image, labelText, cv::Point(box.left - (box.right / 2), box.top - (box.bottom / 2)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
        for (auto& kp : kps) {
            cv::circle(image, cv::Point(kp.x, kp.y), 5, color, -1);
        }
    }
}