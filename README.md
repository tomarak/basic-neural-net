# A Simple Neural Network
This is a simple neural network that makes predictions on 2d arrays of data.

# Example Usage
```python
import numpy
from neural import NeuralNetwork


input_data = numpy.array(([2, 9], [1, 5], [3, 6]), dtype=float)
output_data = numpy.array(([92], [86], [89]), dtype=float)

input_data = input_data/numpy.amax(input_data, axis=0)
output_data = output_data/100

net = NeuralNetwork(2, 3, 1)
net.train_network()
met.saveWeights()
predicted_values = numpy.array(([4,8]), dtype=float)
net.predict(predicted_values)
```
```
Predicted data based on trained weights:
Input (scaled):
[ 0.5  1. ]
Output:
[ 0.90840974]
```


# Future
- [ ] Allow data streams/CSV as input
- [ ] Classification
- [ ] Linear Regression
