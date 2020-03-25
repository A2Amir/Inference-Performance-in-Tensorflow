# 1. Introduction
 In this Repo, I am going to implement various optimizations such as fusion, quantization and machine code compilation to accelerate model prediction then I will walk through how to take advantage of these techniques in Tensorflow. 

The first question you might have is, "Why even bother? Are neural networks fast enough already?" . The answer depends on the application. If we were classifying whether something is a hotdog or not a hotdog, we'd be fine with out of the box inference speeds. 
But things like autonomous vehicles are more demanding than just identifying hotdogs. The perception system in the vehicle has to make predictions in real time. We can't use cloud computing resources due to the latency. So all of these predictions must take place on the vehicle computational hardware. Squeezing every last bit of performance out of the hardware is crucial. 


For example, semantic segmentation is computationally intensive. It's not uncommon for this pipeline to put out four to seven frame per second, while bounding boxes such as YOLO run at 40 to 90 FPS.

<p align="right">
<img src="./imgs/1.png" width="500" height="350"/>
<p align="right">
 
Semantic segmentation has advantages over bounding boxes obviously. But in its current state, the segmentation system is simply not fast enough. However, with various optimizations, we can increase the performance of this system by three to five times. I will cover fusion, quantization and graph to binary optimizations next. 

# 2. Freezing Graphs

Prior to applying any optimizations I will want to freeze the TensorFlow Graph such that it's self-contained in a single [protobuf](https://developers.google.com/protocol-buffers/) file. Freezing the graph is the process of converting TensorFlow variables into constants. During inference, variables become unnecessary since the values they store don’t change. I might as well convert them to constants, which the computer can work with faster.

Additional benefits of freezing the graph are:

* Unnecessary nodes related to training are removed
* The model can be contained entirely in one protobuf file (weights and graph definition)
* Simpler graph structure
* Easier to deploy (due to everything being in one file)

For these reasons, freezing the graph is commonly the first transform engineers execute when optimizing their network for inference. I I will use [the weights of a trained model](https://github.com/A2Amir/Inference-Performance-in-Tensorflow/blob/master/Code/Create_save_weights.ipynb) as the practical example but you can use the weights of your own model.

 ## 2.1. Freezing The Graph Tools
 Here are examples of the freeze graph tools:

* The original freeze_graph function provided by TF is installed in your bin dir and can be called directly if you used PIP to install TF. If  you can not find it please use **pip show tensorflow** to find the path, the tensorflow is installed then navigate to /python/tools or call is directly as below in jupyter notebook.

   ~~~python
   from tensorflow.python.tools.freeze_graph import freeze_graph
   ~~~
   ~/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \

    --input_graph=base_graph.pb \
    --input_checkpoint=ckpt \
    --input_binary=true \
    --output_graph=frozen_graph.pb \
    --output_node_names=Softmax

  The freeze_graph function requires five inputs:
  
  *	The input graph input_graph saved in protobuf format.
  *	The input graph checkpoint, input_checkpoint.
  *	input_binary denotes whether the input graph is a binary file. Set this to true if the input is a .pb file instead of .pbtxt.
  *	The name of the output graph, output_graph, i.e. the frozen graph.
  *	The names of the output nodes. **It’s a good idea in general to name key nodes in the graph and these names will come in handy when using these tools as well.**
  
  The result is saved in the frozen_graph.pb file. 

 * I provide a slightly different version which is simpler and that I found handy. (see this [code](https://github.com/A2Amir/Inference-Performance-in-Tensorflow/blob/master/Code/freez_graph.ipynb)).


# 3. Using freezed Graphs

The [load_graph method](https://github.com/A2Amir/Inference-Performance-in-Tensorflow/blob/master/Code/load_froozen_graph.ipynb) takes a binary protobuf file as input and returns the graph. When the graph is loaded I can take list of operations to make prediction. Check this exercise to get more familiar with using a binary protobuf file to make prediction.

# 4. Graph Transforms

A TensorFlow model is defined as a static graph through which data flows. Graphs are versatile data structures that can be mutated in various ways. TensorFlow takes advantage of this through graph transforms. A transform takes a graph as input, alters it, and returns a new graph as output. Note the original graph used as input is not mutated in place, so remains unaltered. A detailed discussion of many available transforms and how to apply them is found [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms/#introduction). While this information is not required for this lesson, it is worth a read to become familiar with the topic.

<p align="right">
<img src="./imgs/2.png" width="500" height="200"/>
<p align="right">
 
Several transforms can be chained together, typically this is done with a theme in mind. For example, I might want to reduce the graph size, optimize it for inference, create an 8-bit version of the graph, etc. In the following sections I will discuss two sequences of transforms:

1.	Optimizing for Inference

2.	Performing 8-bit Calculations


 ## 4.1. Optimizing for Inference
 
 ### Fusion

Before starting Optimizing for Inference you should have some Information about fusion. The fusion reduces the number of operations and accelerates the data passing through the graph. Consider a three layer pipeline (see gif below): batch normalization, feeding into a Relu, feeding into a convolution. 

The implementations above require each layer to store temporary tensors. I can fuse all three operations together and avoid storing all these extra tensors. Even better, the fuse operation only execute one kernel on the GPU instead of three. 

<p align="right">
<img src="./imgs/1.gif" width="500" height="300"/>
<p align="right">

Each time a kernel is called, there is an overhead. Fusing kernels limits the overhead so the overall applications runs faster. Fusing saves both memory and time. 


Fusing could be beneficial in training as well as inference. The trade-off is that fusing reduces the flexibility of the network. During training, I might want to preserve the flexibility of the model in case I want to add or remove layers or transfer part of the network. By the time I get to inference, I am no longer changing the network architecture, so I can fuse operations more aggressively. 


It's important to know I could do fusing manually by coding up a single kernel that performs the three fuse operations together. However, the compiler is capable of doing this on its own, allowing me to write understandable code and still reap performance benefits. 
I can automate this process using an optimizer that will fuse common layers together. This allow me to write easier to understand code and manipulate it, while the final version after the optimization, will have all the performance advantages by applying tricks like fusion automatically. 
 
 ### Optimizing for Inference

Once the graph is frozen there are a variety of transformations that can be performed; dependent on what I wish to achieve. TensorFlow    has packaged up some inference optimizations in a tool aptly called optimize_for_inference.**
 
 optimize_for_inference does the following:
 
 * Removes training-specific and debug-specific nodes
 * Fuses common operations
 * Removes entire sections of the graph that are never reached
 
 Here’s how it can be used:
 
 ~/tensorflow/bazel-bin/tensorflow/python/tools/optimize_for_inference \
 
--input=frozen_graph.pb \
--output=optimized_graph.pb \
--frozen_graph=True \
--input_names=image_input \
--output_names=Softmax

I'll use the graph I just froze as the input graph. output is the name of the output graph; I’ll be creative and call it optimized_graph.pb. 

The optimize_for_inference tool works for both frozen and unfrozen graphs, so I have to specify whether the graph is already frozen or not with frozen_graph.

input_names and output_names are the names of the input and output nodes respectively. As the option names suggest, there can be more than one input or output node, separated by commas.

Let’s take a look at the other way to use  optimize for inference function in the jupyter note book. Check [this exercise](https://github.com/A2Amir/Inference-Performance-in-Tensorflow/blob/master/Code/OptimizingForInference.ipynb) to get more familiar with this function






 
