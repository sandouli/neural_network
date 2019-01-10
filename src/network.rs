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

    pub fn feed_forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut layer_result = input.clone();
        for mut layer in &mut self.layers {
            layer.calculate_activities(&layer_result);
            layer_result = layer.activities.clone();    // TODO : optimize so no need to clone every layer_result
        }

        layer_result
    }

    pub fn train(&mut self, training_set: &mut Array2<f64>, expected_result: Array2<f64>, objective_function: Objective, batch_size: usize, learning_rate: f64) {
        assert_eq!(training_set.rows(), expected_result.rows(), "Training set should have same amount of rows as expected results");
        assert!(batch_size > 0, "Batch size must be greater than zero");

        let mut i = 0;

        while i < training_set.rows() {

            let current_max_row = training_set.rows().min(batch_size + i);

            let mut data = training_set.slice(s![i..current_max_row, ..]).to_owned();
            let mut expected_result_slice = expected_result.slice(s![i..current_max_row, ..]).to_owned();

            let network_result = self.feed_forward(&data);
            assert_eq!(expected_result.cols(), network_result.cols(), "Expected result and actual result do not have the same amount of columns");

            let total_error = objective_function.calculate_error(&network_result, &expected_result_slice);

            self.backpropagation(&data, &network_result, &expected_result_slice, &objective_function, learning_rate);

            println!("Iteration {}; error: {}", i / batch_size, total_error);
            i += batch_size;
        }
    }

    fn backpropagation(&mut self, input: &Array2<f64>, actual: &Array2<f64>, ideal: &Array2<f64>, objective_function: &Objective, learning_rate: f64) {

        let number_of_layers = self.layers.len();
        let mut result = objective_function.compute_derivative(&actual, &ideal);

        for i in (0..number_of_layers).rev() {

            result = self.layers[i].activation_function.compute_loss(&result, &self.layers[i].output);

            let diff_weight = if i == 0 {
                input.t().dot(&result)
            } else {
                self.layers[i - 1].activities.t().dot(&result)
            };

            let mut diff_bias = Array2::<f64>::zeros((1, result.cols()));
            for j in 0..result.cols() {
                diff_bias[[0, j]] = result.slice(s![.., j]).scalar_sum() / result.rows() as f64;
            }


            if i > 0 {
                result = result.dot(&self.layers[i].weights.t());
            }

            //println!("Layer {}", i);
            //println!("Diff weight : {:?}", diff_weight);
            //println!("Diff bias : {:?}", diff_bias);


            // Update weights and bias
            self.layers[i].bias = &self.layers[i].bias + &(&diff_bias * learning_rate);     // TODO : should learning rate for bias be different ?
            self.layers[i].weights = &self.layers[i].weights + &(&diff_weight * learning_rate);
        }
    }

    pub fn get_expected_input(&self, expected_output: &Array2<f64>) -> Array2<f64> {
        let mut result = expected_output.clone();

        for i in (0..self.layers.len()).rev() {
            result = self.layers[i].activation_function.compute_reverse(&result);
            result = result - &self.layers[i].bias;
            result = result.dot(&self.layers[i].weights.t());
        }

        result
    }
}