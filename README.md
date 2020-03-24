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


# 2. Using freezed Graphs

The [load_graph method](https://github.com/A2Amir/Inference-Performance-in-Tensorflow/blob/master/Code/load_froozen_graph.ipynb) takes a binary protobuf file as input and returns the graph. When the graph is loaded I can take list of operations to make prediction. Check this exercise to get more familiar with using a binary protobuf file to make prediction.

