use rand::distributions::Range;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use activation::Activation;


pub struct Layer {
    neurons: usize,
    weights: Array2<f64>,
    pub activities: Array2<f64>,
    activation_function: Activation,
}

impl Layer {
    pub fn new(neurons: usize, inputs: usize, activation_function: Activation) -> Self {

        // Create matrix with random weights and add one row for the bias weight
        let mut weights = Array2::<f64>::random((inputs + 1, neurons), Range::new(0.0, 1.0));

        // Initialize the last row with the bias weight (equal or close to zero)
        let bias_weight_value = 0.1;    // TODO : should be parameterized ?
        weights.slice_mut(s![-1, ..]).fill(bias_weight_value);


        Self {
            neurons,
            weights,
            activities: Array2::<f64>::zeros((1, 1)),
            activation_function,
        }
    }

    pub fn calculate_activities(&mut self, input: &Array2<f64>) -> &Array2<f64> {

        // Get input data and add the bias node to each row
        let mut input_with_bias = Array2::<f64>::ones((input.rows(), input.cols() + 1));
        input_with_bias.slice_mut(s![.., ..-1]).assign(&input);

        // Compute matrix calculation between input and weights
        let mut output = input_with_bias.dot(&self.weights);

        // Apply activation function
        self.activities = self.activation_function.compute(output);

        // Return layer result
        &self.activities
    }
}