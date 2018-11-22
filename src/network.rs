use ndarray::{Array2};

use layer::Layer;
use activation::Activation;
use objective::Objective;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers,
        }
    }

    pub fn feed_forward(&mut self, input: Array2<f64>) -> Array2<f64> {
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
