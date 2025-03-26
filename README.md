# Neural Network from scratch! 
I started learning about neural networks (NN) in November 2024, then continued with convolutional neural networks (CNN) in February 2025. I had to pause my studies in January and February due to work and personal matters. By the end of March, I had completed a simple construction of NN and CNN by learning how to build them from scratch. Just like when I studied more traditional machine learning algorithms (which I previously posted in other repository), in this phase of learning NN and CNN, I "observed" code from various sources and tried modifying it.  

In short, the most challenging part of understanding both is the backpropagation process and its mathematics—how the learning process is reevaluated when predictions do not match factual output. The hardest part is understanding the chain rule in calculating derivatives. Complicated? In other words, a predictions is generated from chains of information (forward), and when the prediction is incorrect, a backward process must be performed to adjust the parameters at each information chain.  

However, understanding derivatives and the chain rule is not enough, as this process can also lead to numerical instability, causing gradient calculations to become extremely large or small (exploding or vanishing gradients). Simply put, the complexity of both the prediction (forward) and evaluation (backward) processes can become so intricate that the model ends up learning very poorly or not learning at all. 

Another equally exhausting challenge is understanding dimensions or shapes. Both NN and CNN involve layers that perform matrix operations. Many times, the code I wrote resulted in matrices with mismatched dimensions, leading to errors when performing multiplication or convolution. Visualizing the dimensions or shapes in forward propagation is relatively easy. The real challenge is when trying to picture them during the "reverse journey" of backpropagation. "Wait! Why does this need to be transposed?" is a thought that often pops into my mind.  
  
As I mentioned, the following neural network from scratch is a modified version of code originally created by lionelmessi6410. The modifications I made include:  
1. Increasing the layer size from one hidden layer to three hidden layers, making a total of five layers.  
2. Adding methods such as `apply_dropout` and `apply_gradient_clipping` to implement dropout and gradient clipping.  
3. Adding Lasso and Ridge regularization in the `cross_entropy_loss` and `backward` methods to apply penalties on the gradients.  
4. Placing the `train` method outside the `Neural_Network` class and constructing batch data using a different approach.

Enjoy! 
