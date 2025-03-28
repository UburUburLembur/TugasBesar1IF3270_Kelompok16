import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import viznet
import os

class NeuralNetwork:
  def __init__(self, input_size, hidden_layers, output_size, activations, loss_function, weight_init_methods, weight_init_params):
    self.activations = [getattr(self, act) for act in activations] 
    self.activation_derivatives = [getattr(self, act + "_derivative") for act in activations] 
    self.loss_function = getattr(self, loss_function) 
    self.loss_derivative = getattr(self, loss_function + "_derivative") 
    self.history = {"train_loss": [], "val_loss": []}

    layer_sizes = [input_size] + hidden_layers + [output_size]
    self.weights = [self.initialize_weights((layer_sizes[i], layer_sizes[i + 1]), method=weight_init_methods[i], **weight_init_params[i]) for i in range(len(layer_sizes) - 1)]
    self.biases = [self.initialize_weights((1, layer_sizes[i + 1]), method=weight_init_methods[i], **weight_init_params[i]) for i in range(len(layer_sizes) - 1)]
    self.weight_gradients = [np.zeros_like(w) for w in self.weights]


  def forward(self, x):
    activations = [x]
    for w, b, act in zip(self.weights, self.biases, self.activations):
      x = act(np.dot(x, w) + b)
      #print(x.shape)
      activations.append(x)
    return activations

  def backward(self, activations, y, learning_rate):
    if self.loss_function == self.categorical_cross_entropy and self.activations[-1] == self.softmax:
      deltas = [self.categorical_cross_entropy_derivative_softmax(y, activations[-1])]
    else:
      deltas = [self.loss_derivative(y, activations[-1]) * self.activation_derivatives[-1](activations[-1])]

    for i in range(len(self.weights) - 1, 0, -1):
      deltas.insert(0, np.dot(deltas[0], self.weights[i].T) * self.activation_derivatives[i](activations[i]))

    for i in range(len(self.weights)):
      self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i]) / len(y)
      self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0)
      self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0)

  def train(self, X, y, batch_size=32, learning_rate=0.1, epochs=100, verbose=1, X_val=None, y_val=None):
    t = epochs // 10 if epochs > 10 else 1
    for epoch in tqdm(range(epochs), desc="Training Progress", disable=not verbose):
      loss = 0
      indices = np.random.permutation(len(X))
      X, y = X[indices], y[indices]


      for i in range(0, len(X), batch_size):
        X_batch, y_batch = X[i:min(i + batch_size, len(X))], y[i:min(i + batch_size, len(y))]
        activations = self.forward(X_batch)
        loss += np.mean(self.loss_function(y_batch, activations[-1]))
        self.backward(activations, y_batch, learning_rate)

      train_loss = loss / (len(X) / batch_size)
      self.history["train_loss"].append(train_loss)

      if X_val is not None and y_val is not None:
        val_loss = np.mean([self.loss_function(y_val[i], self.forward(X_val[i])[-1]) for i in range(len(X_val))])
        self.history["val_loss"].append(val_loss)

      if verbose == 1:
        if epoch % t == 0:
          print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss if X_val is not None else 'N/A'}")

  def predict(self, X):
    return self.forward(X)[-1]

  def plot_loss(self, title = "Fungsi Loss"):
    plt.plot(self.history["train_loss"], label="Train Loss")
    if self.history["val_loss"]:
      plt.plot(self.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title) if title else None
    plt.legend()
    plt.show()

  def initialize_weights(self, shape, method="random_uniform", **kwargs):
    if method == "zero":
      return np.zeros(shape)
    elif method == "random_uniform":
      np.random.seed(kwargs.get("seed", None))
      return np.random.uniform(kwargs.get("lower", -1), kwargs.get("upper", 1), shape)
    elif method == "random_normal":
      np.random.seed(kwargs.get("seed", None))
      return np.random.normal(kwargs.get("mean", 0), kwargs.get("variance", 1), shape)
    elif method == "he":
      np.random.seed(kwargs.get("seed", None))
      return np.random.normal(0, np.sqrt(2 / shape[0]), shape)
    elif method == "xavier":
      np.random.seed(kwargs.get("seed", None))
      return np.random.normal(0, np.sqrt(1 / shape[0]), shape)
    else:
      raise ValueError("Unknown initialization method")

  def mse_derivative(self, y_true, y_pred):
    return - 2 * (y_pred - y_true)

  def binary_cross_entropy_derivative(self, y_true, y_pred):
    epsilon = 1e-8
    return (y_pred - y_true) / ((y_pred * (1 - y_pred) + epsilon))

  def categorical_cross_entropy_derivative_softmax(self, y_true, y_pred):
    return y_pred - y_true

  def categorical_cross_entropy_derivative(self, y_true, y_pred):
    return - y_true / y_pred

  def linear_derivative(self, o):
    return np.ones_like(o)

  def relu_derivative(self, o):
    return np.where(o > 0, 1, 0)

  def sigmoid_derivative(self, o):
    return o * (1 - o)

  def hyperbolic_tangent_derivative(self, o):
    return (2/(np.exp(o) + np.exp(-o)))**2

  def softmax_derivative(self, o):
    return self.softmax(o) * (1 - self.softmax(o))

  def mse(self, out, target):
    return np.mean(np.square(out - target))

  def binary_cross_entropy(self, out, target):
    return -np.mean(target * np.log(out + 1e-8) + (1 - target) * np.log(1 - out + 1e-8))

  def categorical_cross_entropy(self, out, target):
    return -np.mean(np.sum(target * np.log(out + 1e-9), axis=1))

  def linear(self, net):
    return net

  def relu(self, net):
    return np.maximum(0, net)

  def sigmoid(self, net):
    return 1 / (1 + np.exp(-net))

  def hyperbolic_tangent(self, net):
    return np.tanh(net)

  def softmax(self, net):
    e_net = np.exp(net - np.max(net, axis=1, keepdims=True))
    return e_net / np.sum(e_net, axis=1, keepdims=True)

  def plot_weight_distribution(self, layers):
    plt.figure(figsize=(12, 4 * len(layers)))

    for i, layer_idx in enumerate(layers, start=1):
      w = self.weights[layer_idx]
      w_flat = w.flatten()

      plt.subplot(len(layers), 1, i)
      plt.hist(w_flat, bins=30, alpha=0.7, color='blue')
      plt.title(f"Weight Distribution - Layer {layer_idx}")
      plt.xlabel("Weight Value")
      plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

  def display_graph(self):
    plt.figure(figsize=(24, 16))

    grid = viznet.Grid((150.0, 80.0))
    brush = viznet.NodeBrush('basic', size=20, color='lightblue')
    edge_brush = viznet.EdgeBrush('->', lw=0.1, color='black')

    NODE_FONT_SIZE = 7
    EDGE_FONT_SIZE = 8

    orig_layer_sizes = [self.weights[0].shape[0]] + [w.shape[1] for w in self.weights]
    num_layers = len(orig_layer_sizes)

    layers_nodes = []

    for layer_index in range(num_layers):
      nodes = []
      if layer_index < num_layers - 1:
        bias_node = brush >> grid[layer_index, 0]
        bias_val = self.biases[layer_index - 1][0, 0] if layer_index > 0 else self.biases[0][0, 0]
        bias_node.text(f"b_{layer_index}", fontsize=NODE_FONT_SIZE)
        nodes.append(bias_node)
        for neuron_index in range(orig_layer_sizes[layer_index]):
          node = brush >> grid[layer_index, neuron_index + 1]
          if layer_index == 0:
            node.text(f"x_{neuron_index+1}", fontsize=NODE_FONT_SIZE)
          else:
            node.text(f"N_{layer_index}_{neuron_index+1}", fontsize=NODE_FONT_SIZE)
          nodes.append(node)
      else:
        for neuron_index in range(orig_layer_sizes[layer_index]):
          node = brush >> grid[layer_index, neuron_index]
          node.text(f"Out_{neuron_index+1}", fontsize=NODE_FONT_SIZE)
          nodes.append(node)
      
      layers_nodes.append(nodes)

    for layer_index in range(len(layers_nodes) - 1):
      weight_matrix = self.weights[layer_index]
      grad_matrix = self.weight_gradients[layer_index]
      current_nodes = layers_nodes[layer_index]
      next_nodes = layers_nodes[layer_index + 1]

      start_index = 1  if layer_index < len(layers_nodes) - 1 else 0
      next_start = 1 if (layer_index + 1 < len(layers_nodes) - 1) else 0

      for m, node_from in enumerate(current_nodes[start_index:]):
        for j, node_to in enumerate(next_nodes[next_start:]):
          w_val = weight_matrix[m, j]
          g_val = grad_matrix[m, j]
          edge = edge_brush >> (node_from, node_to)
          edge.text(f"w:{w_val:.2f}", position="left", fontsize=EDGE_FONT_SIZE)

          edge.text(f"g:{g_val:.2f}", position="right", fontsize=EDGE_FONT_SIZE)


      if layer_index < num_layers - 1:
        bias_node = current_nodes[0]  # Bias node berada di indeks 0
        next_start = 1 if layer_index + 1 < num_layers - 1 else 0
        for j, node_to in enumerate(next_nodes[next_start:]):
          bias_weight = self.biases[layer_index][0, j]
          edge = edge_brush >> (bias_node, node_to)
          edge.text(f"b:{bias_weight:.2f}", "center", fontsize=EDGE_FONT_SIZE)
    
    plt.title("Representasi Graf")
    plt.show()


  def plot_gradient_distribution(self, layers):
    plt.figure(figsize=(12, 4 * len(layers)))

    for i, layer_idx in enumerate(layers, start=1):
      g = self.weight_gradients[layer_idx]
      g_flat = g.flatten()

      plt.subplot(len(layers), 1, i)
      plt.hist(g_flat, bins=30, alpha=0.7, color='red')
      plt.title(f"Gradient Distribution - Layer {layer_idx}")
      plt.xlabel("Gradient Value")
      plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
  
  def save_model(self, filename="ann_model.npz"):
    np.savez(filename, **{f"weights_{i}": w for i, w in enumerate(self.weights)},
                      **{f"biases_{i}": b for i, b in enumerate(self.biases)})
    print(f"Model saved to {filename}")

  def load_model(self, filename="ann_model.npz"):
    if os.path.exists(filename):
      data = np.load(filename, allow_pickle=True)
      self.weights = [data[f"weights_{i}"] for i in range(len(self.weights))]
      self.biases = [data[f"biases_{i}"] for i in range(len(self.biases))]
      print(f"Model loaded from {filename}")
    else:
      print("No saved model found!")
