
use network::NeuralNetwork;
use layer::Layer;
use activation::Activation;


pub struct NeuralNetworkBuilder {
    last_layer_outputs: usize,
    layers: Vec<Layer>,
}

impl NeuralNetworkBuilder {
    pub fn new(inputs: usize) -> Self {
        Self {
            last_layer_outputs: inputs,
            layers: Vec::new(),
        }
    }

    pub fn layer(mut self, neurons: usize, activation_function: Activation) -> Self {
        self.layers.push(Layer::new(neurons, self.last_layer_outputs, activation_function));
        self.last_layer_outputs = neurons;
        self
    }

    pub fn build(mut self) -> NeuralNetwork {
        assert!(!self.layers.is_empty(), "No layers defined");
        NeuralNetwork::new(self.layers)
    }

}

