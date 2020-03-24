# 1. Introduction
 In this Repo, I am going to implement various optimizations such as fusion, quantization and machine code compilation to accelerate model prediction then I will walk through how to take advantage of these techniques in Tensorflow. 

The first question you might have is, "Why even bother? Are neural networks fast enough already?" . The answer depends on the application. If we were classifying whether something is a hotdog or not a hotdog, we'd be fine with out of the box inference speeds. 
But autonomous vehicles are more demanding than just identifying hotdogs. The perception system in the vehicle has to make predictions in real time. We can't use cloud computing resources due to the latency. So all of these predictions must take place on the vehicle computational hardware. Squeezing every last bit of performance out of the hardware is crucial. 


For example, semantic segmentation is computationally intensive. It's not uncommon for this pipeline to put out four to seven frame per second, while bounding boxes such as YOLO run at 40 to 90 FPS.

<p align="right">
<img src="./imgs/1.png" width="500" height="350"/>
<p align="right">
 
