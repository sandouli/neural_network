use rand::distributions::Range;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use activation::Activation;


pub struct Layer {
    neurons: usize,
    pub weights: Array2<f64>,
    pub bias: Array2<f64>,
    pub output: Array2<f64>,
    pub activities: Array2<f64>,
    pub activation_function: Activation,
}

impl Layer {
    pub fn new(neurons: usize, inputs: usize, activation_function: Activation) -> Self {

        // Create weights matrix with random values
        let weights = Array2::<f64>::random((inputs, neurons), Range::new(0.0, 1.0));

        // Create bias matrix with constant value
        let mut bias = Array2::<f64>::zeros((1, neurons));
        let bias_default_value = 0.1;    // TODO : determine default value for bias
        bias.fill(bias_default_value);


        Self {
            neurons,
            weights,
            bias,
            output: Array2::<f64>::zeros((1, 1)),
            activities: Array2::<f64>::zeros((1, 1)),
            activation_function,
        }
    }

    pub fn calculate_activities(&mut self, input: &Array2<f64>) {

        // Compute matrix calculation between input and weights
        self.output = input.dot(&self.weights) + &self.bias;

        // Apply activation function
        self.activities = self.activation_function.compute(&self.output);
    }
}