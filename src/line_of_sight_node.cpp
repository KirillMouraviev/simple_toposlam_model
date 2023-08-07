#include <iostream>
#include "ros/ros.h"
#include "geometry_msgs/Point32.h"
#include "geometry_msgs/Pose.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Path.h"
#include "std_msgs/Float32MultiArray.h"

#include "simple_toposlam_model/line_of_sight.hpp"
#include "simple_toposlam_model/map.hpp"

#define DEFAULT_AGENT_RADIUS 0.2

Map map;
bool map_is_updated = false;
ros::Publisher pub;
float agent_radius;
LineOfSight visChecker;

void mapCallback(const nav_msgs::OccupancyGrid& msg)
{
    ROS_INFO("Map recieved!");
    Point pos = Point(msg.info.origin.position.x, msg.info.origin.position.y);
    Quaternion orient = {msg.info.origin.orientation.x, msg.info.origin.orientation.y, msg.info.origin.orientation.z, msg.info.origin.orientation.w};
    map.Update(msg.info.resolution, msg.info.height, msg.info.width, pos, orient, msg.data);
    map_is_updated = true;
    visChecker.setSize(agent_radius / map.GetCellSize());
}

void taskCallback(const std_msgs::Float32MultiArray& msg)
{
    ROS_INFO("Task received!");
    if (!map_is_updated)
    {
        return;
    }
    Point pt1(msg.data[0], msg.data[1]);
    Point pt2(msg.data[2], msg.data[3]);
    Node n1 = map.GetClosestNode(pt1);
    Node n2 = map.GetClosestNode(pt2);
    ROS_INFO("Check from (%d, %d) to (%d, %d)", n1.i, n1.j, n2.i, n2.j);
    bool in_sight = visChecker.checkLine(n1.i, n1.j, n2.i, n2.j, map);
    std_msgs::Float32MultiArray response_msg;
    response_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    response_msg.layout.dim[0].label = "width";
    response_msg.layout.dim[0].size = 5;
    response_msg.layout.dim[0].stride = 5;
    response_msg.layout.data_offset = 0;
    response_msg.data.push_back(msg.data[0]);
    response_msg.data.push_back(msg.data[1]);
    response_msg.data.push_back(msg.data[2]);
    response_msg.data.push_back(msg.data[3]);
    response_msg.data.push_back(in_sight);
    pub.publish(response_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "line_of_sight_node");
    ros::NodeHandle n;
    agent_radius = DEFAULT_AGENT_RADIUS;
    ros::param::get("~agent_radius", agent_radius);
    visChecker = LineOfSight(agent_radius / map.GetCellSize());

    ros::Subscriber sub1 = n.subscribe("map", 10, mapCallback);
    ros::Subscriber sub2 = n.subscribe("task", 10, taskCallback);

    pub = n.advertise<std_msgs::Float32MultiArray>("response", 1000);
    ros::spin();
}
