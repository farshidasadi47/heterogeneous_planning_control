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
#include <geometry_msgs/msg/point32.h>
// Micro-ros related
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
// Coils custom library
#include "coils.h"
/********* Globals ****************************************************/
#define LED_PIN 13
float field[3]{0};
Coils coil;
// Timer period on nanoseconds
const unsigned int timer_period = RCL_MS_TO_NS(10);  // 100.0 Hz
// ROS2 messages
geometry_msgs__msg__Point32 field_sub;
geometry_msgs__msg__Point32 field_pub;
// Micro-ros node
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;
// Publishers ans subscribers
rcl_publisher_t field_publisher;
rcl_subscription_t field_subscriber;
/********* Function declarations **************************************/
// Micro-ros related.
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}
void main_loop();   // Timed loop code runs in this function.
void error_loop();  // Handles error in initiation.
void field_sub_callback(const void* msgin);  // Magnetic field commands.
void msg_sub_callback(const void* msgin);    // Latency check.
void timer_callback(rcl_timer_t * timer, int64_t last_call_time);  // Timed loop.
/********* Initializations ********************************************/
void setup() {
    set_microros_transports();  // Do not know what this is.
    // Set LED on, indicating normal operation.
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);
    // Initialize coils.
    coil.initialize();
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
    const rosidl_message_type_support_t* type_support_field =
                         ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32);
    // Creates a best_effort rcl publisher
    rc = rclc_publisher_init_best_effort(&field_publisher, &node, 
                                         type_support_field, "arduino_field_fb");
    RCCHECK(rc);
    // Set up subscribers.
    rc = rclc_subscription_init_best_effort(&field_subscriber,&node,
                                        type_support_field, "arduino_field_cmd");
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
    rc = rclc_executor_add_subscription(&executor, &field_subscriber, &field_sub,
                                        &field_sub_callback, ON_NEW_DATA);
    RCCHECK(rc);
    // Add timer to the executor
    rc = rclc_executor_add_timer(&executor, &timer);
    RCCHECK(rc);
    //
    field_sub.x = 0;
    field_sub.y = 0;
    field_sub.z = 0;
}
/********* Main loop **************************************************/
void loop() {
    delay(100);
    RCSOFTCHECK(rclc_executor_spin(&executor));
}
/********* Function definitions ***************************************/
void main_loop(){
    // All the process should be programmed here.
    // Toggle PIN13, to check loop frequency by oscilloscope
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    // Update field command feedback
    field_pub.x = field_sub.x;
    field_pub.y = field_sub.y;
    field_pub.z = field_sub.z;
    // Update coil voltages.
    coil.set_magnetic_field(field_sub.x, field_sub.y, field_sub.z);
    // Publish the latest latency check variable.
    RCSOFTCHECK(rcl_publish(&field_publisher, &field_pub, NULL));
}

void error_loop(){
    // Blinks the LED, indicating error. Micropcs should be reset.
    while(1){
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        delay(100);
    }
}

void field_sub_callback(const void* msgin){
    // Subscribes to magnetic field command.
    // x, y, and z are theta, alpha, and power percentage.
    const geometry_msgs__msg__Point32* msg = (const geometry_msgs__msg__Point32*) msgin;
    field_sub.x = msg->x;
    field_sub.y = msg->y;
    field_sub.z = msg->z;
}

void timer_callback(rcl_timer_t * timer, int64_t last_call_time){  
    // This will be executed as a timed loop.
    RCLC_UNUSED(last_call_time);
    if (timer != NULL) {
        main_loop();
    }
}
