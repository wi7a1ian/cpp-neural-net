# TODO
- [ ] make it more declarative and generic (functional, lambdas, static polymorphism)
- [ ] change to fluent api + implify like in `torch`
  ```cpp
  struct Model : torch::nn::Module {
    Model() {
        in = register_module("in",torch::nn::Linear(8,64));
        h = register_module("h",torch::nn::Linear(64,64));
        out = register_module("out",torch::nn::Linear(64,1));
    }
    torch::Tensor forward(torch::Tensor X){
        // let's pass relu 
        X = torch::relu(in->forward(X));
        X = torch::relu(h->forward(X));
        X = torch::sigmoid(out->forward(X));
        
        // return the output
        return X;
    }
    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};
  };
  ```
- [ ] rename `Backpropagation` to `GradientDescent` coz one is the technique finding (partial) error and updating the weights and latter one is an implementation of it
- [ ] activation functions should be selected per layer, not per network
- [ ] use sigmoidn activation function for problems like predicting probabilities (last layer, multiple output neurons)
- [ ] add softmax activation function for problems like probability distribution output (last layer, single output neuron)
- [ ] add relu activation function for all the hidden layers (also add leaky relu)
- [ ] for a binary classification problem, if the model outputs a probability, we should use the binary cross-entrophy loss function
- [ ] for multi-class classification we use categorical cross-entrophy loss function
- [ ] make it more explicit that for regression problem, if the model outputs predicted value, we should use means square error
- [ ] add adagrad and adam optimization functions
- [ ] update gradient descent to be able to work with whole trainign set, one item or part of (batch grad desc, stochastic, mini-batch)
- [ ] add convolution neural network
