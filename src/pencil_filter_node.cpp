#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "gate_fusion/pencil_filter.hpp"

class PencilFilterNode {
public:
    PencilFilterNode()
        : nh_(), it_(nh_), pencil_filter_(2, cv::MORPH_ELLIPSE) {
    
    if (nh_.getParam("image_topic_name", topic_name_) != true) {
      ROS_ERROR("Fail to get image topic name!");
      exit(1);
    }

        image_sub_ = it_.subscribe(topic_name_, 1, &PencilFilterNode::imageCallback, this);
        image_pub_ = it_.advertise(topic_name_ + "/pencil", 1);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat processed_image = pencil_filter_.apply(cv_ptr->image);

        sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(msg->header, "bgr8", processed_image).toImageMsg();
        image_pub_.publish(output_msg);
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    PencilFilter pencil_filter_;
    std::string topic_name_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pencil_filter_node");
    PencilFilterNode pencil_filter_node;
    ros::spin();
    return 0;
}
