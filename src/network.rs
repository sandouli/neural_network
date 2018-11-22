use ndarray::{Array2};

use layer::Layer;
use activation::Activation;
use objective::Objective;

pub struct NeuralNetwork {
    last_layer_outputs: usize,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(inputs: usize) -> Self {
        Self {
            last_layer_outputs: inputs,
            layers: Vec::new(),
        }
    }

    pub fn layer(&mut self, neurons: usize, activation: Activation) {
        self.layers.push(Layer::new(neurons, self.last_layer_outputs, activation));
        self.last_layer_outputs = neurons;
    }

    pub fn feed_forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        assert!(!self.layers.is_empty(), "No layers defined");
        let mut layer_result = input;
        for mut layer in &mut self.layers {
            layer.calculate_activities(&layer_result);
            layer_result = layer.activities.clone();    // TODO : optimize so no need to clone every layer_result
        }

        layer_result
    }

    pub fn train(&mut self, training_set: Array2<f64>, expected_result: Array2<f64>, objective_function: Objective) {
        // Training set contains input data


        // Stochastic gradient descent -> Mini batches of training set to compare to expected results instead of the whole batch ?
    }

}
