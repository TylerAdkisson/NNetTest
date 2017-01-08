# Neural Network Test Project #

This is a collection of code from when I decided to learn how artificial neural networks worked.  It's nothing fancy, just code that allows 
you to build simple neural networks capable of learning simple non-linear functions (like XOR), and even _very_ basic image processing.
Neuron activation functions include

* Hyperbolic tangent.  Values are squished into the range -1.0..+1.0
* Rectifier (ReLU).  Values are in the range 0.0..+1.0
* 'Leaky' rectifier (LeakyReLU).  Works the same as the normal ReLU, except allows a small value in the negative range.

Backpropagation is the only implemented training algorithm, with the following learning methods:

* Stochastic gradient descent with momentum and weight decay
* Adagrad

Note: The code is a bit of a mess in places.
