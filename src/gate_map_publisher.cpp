#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include "drolib/rotation/rotation_utils.h"

void load_gatemap(const std::string& filepath,
                  std::vector<Eigen::Vector3d>& gate_map_p,
                  std::vector<Eigen::Quaterniond>& gate_map_q) {
  YAML::Node config = YAML::LoadFile(filepath);

  // Check if the YAML file loaded correctly
  if (!config) {
    std::cerr << "Error loading YAML file: " << filepath << std::endl;
    return;
  }

  // Iterate through all nodes in the YAML file
  for (const auto& node : config) {
    // Ensure that the node is a map and contains the "position" key
    if (node.second.IsMap() && node.second["position"]) {
      const YAML::Node& positionNode = node.second["position"];
      if (positionNode.size() == 3) {
        double x = positionNode[0].as<double>();
        double y = positionNode[1].as<double>();
        double z = positionNode[2].as<double>();

        Eigen::Vector3d position(x, y, z);
        gate_map_p.push_back(position);
      }
    }

    if (node.second.IsMap() && node.second["rpy"]) {
      const YAML::Node& rpyNode = node.second["rpy"];
      if (rpyNode.size() == 3) {
        double r = rpyNode[0].as<double>();
        double p = rpyNode[1].as<double>();
        double y = rpyNode[2].as<double>();

        Eigen::Quaterniond q(drolib::eulerAnglesRPYToQuaternion(Eigen::Vector3d(r, p, y)));
        gate_map_q.push_back(q);
      }
    }
  }
    std::cout <<"Loaded " << gate_map_p.size() << " gates." << std::endl;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "gate_map_publisher");
  ros::NodeHandle nh;

  std::string gate_map_path;

  if (nh.getParam("gate_map_path", gate_map_path) != true) {
    ROS_ERROR("Fail to get param file path!");
    exit(1);
  }

  std::vector<Eigen::Vector3d> gate_map_p;
  std::vector<Eigen::Quaterniond> gate_map_q;
  load_gatemap(gate_map_path, gate_map_p, gate_map_q);

  geometry_msgs::Vector3 dimensions;

  dimensions.x = 0.01;
  dimensions.y = 1.8288;
  dimensions.z = 1.8288;

  jsk_recognition_msgs::BoundingBoxArray gate_msg;
  gate_msg.header.stamp = ros::Time::now();
  gate_msg.header.frame_id = "world";
  for (int i = 0; i < static_cast<int>(gate_map_p.size()); i++) {
    jsk_recognition_msgs::BoundingBox gate;
    gate.header.stamp = ros::Time::now();
    gate.header.frame_id = "world";
    gate.pose.position.x = gate_map_p[i](0);
    gate.pose.position.y = gate_map_p[i](1);
    gate.pose.position.z = gate_map_p[i](2);
    gate.pose.orientation.x = gate_map_q[i].x();
    gate.pose.orientation.y = gate_map_q[i].y();
    gate.pose.orientation.z = gate_map_q[i].z();
    gate.pose.orientation.w = gate_map_q[i].w();
    gate.label = i;
    gate.dimensions = dimensions;
    gate_msg.boxes.push_back(gate);
  }

  ros::Publisher gate_map_pub = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("gate_fusion/gate_map", 1000);

  ros::Rate loop_rate(10);

  while (ros::ok())
  {
    gate_map_pub.publish(gate_msg);

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}