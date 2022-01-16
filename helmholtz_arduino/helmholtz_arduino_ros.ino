/***********************************************************************
* This file contains a template program to communicate between arduino
* and computer.
* It communicates with computer via ROS2 micro-ros interface.
* Author: Farshid Asadi, farshidasadi47@yahoo.com
/********* Includes ***************************************************/
// Standards
#include <stdio.h>
// ROS2 related
#include <micro_ros_arduino.h>
#include <std_msgs/msg/int32.h>
// Micro-ros related
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
/********* Globals ****************************************************/
#define LED_PIN 13
#define TIMIG_PIN 12
// Timer period on nanoseconds
const unsigned int timer_period = RCL_MS_TO_NS(10);
// ROS2 messages
std_msgs__msg__Int32 msg_pub;
std_msgs__msg__Int32 msg_sub;
// Micro-ros node
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;
// Publishers ans subscribers
rcl_publisher_t publisher;
rcl_subscription_t subscriber;
/********* Functions **************************************************/
// Micro-ros related.
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}
void main_loop();  // Timed loop code runs in this function.
void error_loop(){
    // Blinks the LED, indicating error. Micropcs should be reset.
    while(1){
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        delay(100);
    }
}
void subscription_callback(const void* msgin){
    const std_msgs__msg__Int32* msg = (const std_msgs__msg__Int32*) msgin;
    msg_sub.data = msg->data;
}
void timer_callback(rcl_timer_t * timer, int64_t last_call_time){  
    // This will be executed as a timed loop.
    RCLC_UNUSED(last_call_time);
    if (timer != NULL) {
        main_loop();
    }
}
/********* Initializations ********************************************/
void setup() {
    set_microros_transports();  // Do not know what this is.
    // Set LED on, indicating normal operation.
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    // Initialize micro-ROS allocator.
    allocator = rcl_get_default_allocator();
    // Initialize support object.
    rcl_ret_t rc = rclc_support_init(&support, 0, NULL, &allocator);
    RCCHECK(rc);
    // Init default node.
    rc = rclc_node_init_default(&node, "arduino", "", &support);
    RCCHECK(rc);
    // Set up publishers.
    // Get message type support
    const rosidl_message_type_support_t* type_support_pub =
                         ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32);
    // Creates a best_effort rcl publisher
    rc = rclc_publisher_init_best_effort(&publisher, &node, 
                                         type_support_pub, "arduino_feedback");
    RCCHECK(rc);
    // Set up subscribers.
    // Get message type support
    const rosidl_message_type_support_t* type_support_sub =
                         ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32);
    // Initialize a reliable subscriber
    rc = rclc_subscription_init_default(&subscriber,&node,
                                        type_support_sub, "arduino_command");
    RCCHECK(rc);
    // Initialize timer object.
    rc = rclc_timer_init_default(&timer,&support,timer_period,timer_callback);
    RCCHECK(rc);
    // Initialize executer
    // total number of handles = #subscriptions + #timers
    unsigned int num_handles = 1 + 1 ;
    rc = rclc_executor_init(&executor,&support.context,num_handles,&allocator);
    RCCHECK(rc);
    // Adding subscribers and timers. Orders matter.
    // Add subscriber to executer.
    rc = rclc_executor_add_subscription(&executor, &subscriber, &msg_sub,
                                        &subscription_callback, ON_NEW_DATA);
    // Add timer to the executor
    rc = rclc_executor_add_timer(&executor, &timer);
    RCCHECK(rc);
    //
    msg_sub.data = 0;
    msg_pub.data = 0;
}
/********* Main loop **************************************************/
void loop() {
    delay(100);
    RCSOFTCHECK(rclc_executor_spin(&executor));
}
void main_loop(){
    // All the process should be programmed here.
    // Input: None.
    // Output: None.
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    msg_pub.data = msg_sub.data;
    RCSOFTCHECK(rcl_publish(&publisher, &msg_pub, NULL));
}