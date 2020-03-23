# TODO
- [ ] make it more declarative and generic (functional, lambdas, static polymorphism)
- [ ] change to fluent api
- [ ] activation funcstions should be selected per layer, not per network
- [ ] add softmax activation function for problems like probability distribution output (last layer)
- [ ] addf relu activation function for all the hidden layers (also add leaky relu)
- [ ] for a binary classification problem, if the model outputs a probability, we should use the binary cross-entrophy loss function
- [ ] for multi-class classification we use categorical cross-entrophy loss function
- [ ] make it more explicit that for regression problem, if the model outputs predicted value, we should use means square error
- [ ] add adagrad and adam optimization functions
- [ ] update gradient descent to be able to work with whole trainign set, one item or part of (batch grad desc, stochastic, mini-batch)
