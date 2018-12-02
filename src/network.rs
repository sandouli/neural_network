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

    pub fn train(&mut self, training_set: &mut Array2<f64>, expected_result: Array2<f64>, objective_function: Objective, batch_size: usize) {
        assert_eq!(training_set.rows(), expected_result.rows(), "Training set should have same amount of rows as expected results");
        assert!(batch_size > 0, "Batch size must be greater than zero");

        let mut data = Array2::<f64>::ones((batch_size, training_set.cols()));
        let mut result = Array2::<f64>::ones((batch_size, expected_result.cols()));

        let mut i = 0;

        while i < training_set.rows() {

            data.slice_mut(s![.., ..]).assign(&training_set.slice(s![i..training_set.rows().min(batch_size + i), ..]));
            result.slice_mut(s![.., ..]).assign(&expected_result.slice(s![i..training_set.rows().min(batch_size + i), ..]));

            let network_result = self.feed_forward(&data);
            assert_eq!(expected_result.cols(), network_result.cols(), "Expected result and actual result do not have the same amount of columns");

            let total_error = objective_function.calculate_error(&network_result, &result);

            self.backpropagation(&data, &network_result, &result, &objective_function);

            println!("Iteration {}; error: {}", i / batch_size, total_error[[0, 0]]);
            i += batch_size;

        }
    }

    fn backpropagation(&mut self, input: &Array2<f64>, actual: &Array2<f64>, ideal: &Array2<f64>, objective_function: &Objective) {
        // Chain rule of last layer :
        // 2 * (activities of last layer - expected of last layer)
        // *
        // derivative_activation_function(output_last_layer)
        // *
        // activities_before_layer
        //

        let base: Array2<f64> = 2.0 * (ideal - actual);

        let number_of_layers = self.layers.len();

        let mut result = base.clone();

        for i in (0..number_of_layers).rev() {

            result = if i == number_of_layers - 1 {
                result * self.layers[i].activation_function.compute_derivative(&self.layers[i].output)
            } else {
                remove_bias_column(&result) * self.layers[i].activation_function.compute_derivative(&self.layers[i].output)
            };

            let diff_weight = if i == 0 {
                add_bias_column(&input).t().dot(&result)
            } else {
                add_bias_column(&self.layers[i - 1].activities).t().dot(&result)
            };

            if i > 0 {
                result = result.dot(&self.layers[i].weights.t());
            }

            let learning_rate = 0.001;   // TODO : add parameter learning_rate to training function

            self.layers[i].weights = &self.layers[i].weights + &(&diff_weight * learning_rate);

        }

    }

}


fn add_bias_column(array: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::<f64>::ones((array.rows(), array.cols() + 1));
    result.slice_mut(s![.., ..-1]).assign(&array);

    result
}

fn remove_bias_column(array: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((array.rows(), array.cols() - 1));
    result.slice_mut(s![.., ..]).assign(&array.slice(s![.., ..-1]));

    result
}