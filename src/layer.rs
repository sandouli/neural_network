use rand::distributions::Range;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use activation::Activation;


pub struct Layer {
    neurons: usize,
    weights: Array2<f64>,
    pub activities: Array2<f64>,
    activation: Activation,
}

impl Layer {
    pub fn new(neurons: usize, inputs: usize, activation: Activation) -> Self {
        Self {
            neurons,
            weights: get_random_weights_and_bias_matrix(inputs, neurons),
            activities: Array2::<f64>::zeros((1, 1)),
            activation,
        }
    }

    pub fn calculate_activities(&mut self, input: &Array2<f64>) -> &Array2<f64> {

        // Get input data and add the bias node to each row
        let mut input_with_bias = Array2::<f64>::ones((input.rows(), input.cols() + 1));
        input_with_bias.slice_mut(s![.., ..-1]).assign(&input);

        // Compute matrix calculation between input and weights
        let mut output = input_with_bias.dot(&self.weights);

        // Apply activation function
        self.activities = match self.activation {
            Activation::TanH => {
                output.map(|v| v.tanh())
            },
            _ => unreachable!(),
        };

        &self.activities

    }
}

fn get_random_weights_and_bias_matrix(rows: usize, columns: usize) -> Array2<f64> {

    // Create matrix with random weights and add one row for the bias weight
    let mut weights = Array2::<f64>::random((rows + 1, columns), Range::new(0.0, 1.0));

    // Initialize the last column of each row with the bias weight (zero or close to zero)
    let bias_weight_value = 0.1;
    weights.slice_mut(s![.., -1]).fill(bias_weight_value);

    weights
}