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

  yoloPub_ = it.advertise(img_topic + "_yolo_viz", 1000);

  model = new YOLO::DeployDet(engine_file_path);
  labels = YOLOSubscriber::generateLabelColorPairs(label_file_path);

}

void YOLOSubscriber::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  double t1 = ros::Time::now().toSec();
  cv_bridge::CvImagePtr BridgePtr;
  BridgePtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  YOLO::Image image_in(BridgePtr->image.data, BridgePtr->image.cols, BridgePtr->image.rows);
  double t2 = ros::Time::now().toSec();
  auto result = model->predict(image_in);
  double t3 = ros::Time::now().toSec();
  YOLOSubscriber::visualize(BridgePtr->image, result, labels);

  sensor_msgs::ImagePtr yolo_msg = cv_bridge::CvImage(msg->header, "bgr8", BridgePtr->image).toImageMsg();
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
        labelColorPairs.emplace_back(label, cv::Scalar(255, 0, 255));
    }
    return labelColorPairs;
}

// Visualize detection results
void YOLOSubscriber::visualize(cv::Mat& image, const YOLO::DetectionResult& result, const std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs) {
    // Define colors for the four corners and edges
    std::vector<cv::Scalar> cornerColors = {
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 255, 0)   // Cyan
    };

    for (size_t i = 0; i < result.num; ++i) {
        const auto& box       = result.boxes[i];
        int         cls       = result.classes[i];
        float       score     = result.scores[i];
        const auto& kps       = result.kps[i];
        const auto& label     = labelColorPairs[cls].first;
        const auto& color     = labelColorPairs[cls].second;
        std::string labelText = label + " " + cv::format("%.2f", score);

        // Draw rectangle and label (optional, currently commented out)
        int      baseLine;
        cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        // cv::rectangle(image, cv::Point(box.x_center - (box.width / 2), box.y_center - (box.height / 2)), cv::Point(box.x_center + (box.width / 2), box.y_center + (box.height / 2)), color, 2, cv::LINE_AA);
        // cv::rectangle(image, cv::Point(box.x_center - (box.width / 2), box.y_center - (box.height / 2) - labelSize.height), cv::Point(box.x_center - (box.width / 2) + labelSize.width, box.y_center - (box.height / 2)), color, -1);
        // cv::putText(image, labelText, cv::Point(box.x_center - (box.width / 2), box.y_center - (box.height / 2)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        // Draw keypoints and connecting lines with different colors
        if (kps.size() == 4) {  // Ensure there are at least 4 keypoints
            // Draw keypoints with different colors
            for (size_t j = 0; j < 4; ++j) {
                cv::circle(image, cv::Point(kps[j].x, kps[j].y), 10, cornerColors[j], -1);
            }

            // Draw lines connecting keypoints with different colors
            for (size_t j = 0; j < 4; ++j) {
                cv::line(image, cv::Point(kps[j].x, kps[j].y), cv::Point(kps[(j+1) % 4].x, kps[(j+1) % 4].y), cornerColors[j], 5, cv::LINE_AA);
            }
        }
    }
}

