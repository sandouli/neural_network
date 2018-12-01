extern crate rand;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;




pub mod builder;
pub mod layer;
pub mod activation;
pub mod objective;
pub mod network;


use rand::distributions::Range;
use ndarray::{Array2, arr2, arr1};
use ndarray_rand::RandomExt;
use builder::NeuralNetworkBuilder;
use activation::Activation;
use objective::Objective;




fn main() {

    test_training_addition_network();

    // TODO : test current implementation with MNIST

}


fn test_training_addition_network() {


    let mut input_data = Array2::<f64>::zeros((10000, 2));
    let mut expected_result = Array2::<f64>::zeros((10000, 1));

    let mut new_input_data: Array2<f64>;
    let mut new_expected_result: Array2<f64>;


    for i in 0..1000 {
        for j in 0..10 {
            input_data.slice_mut(s![i*10 + j, ..]).assign(&arr1(&[i as f64 / 100.0, j as f64 / 100.0]));

            expected_result.slice_mut(s![i*10 + j, ..]).assign(&arr1(&[(i + j) as f64 / 100.0]));

        }
    }

    let mut network = NeuralNetworkBuilder::new(input_data.cols())
        .layer(2, Activation::ReLU)
        .layer(3, Activation::ReLU)
        .layer(4, Activation::ReLU)
        .layer(5, Activation::ReLU)
        .layer(2, Activation::ReLU)
        .layer(1, Activation::Identity)
//        .layer(4, Activation::Sigmoid)
//        .layer(4, Activation::LeakyReLU(0.3))
//        .layer(1, Activation::LeakyReLU(0.3))
//        .layer(1, Activation::Softmax)
        .build();

    println!("Before training : {:?}", network.feed_forward(&arr2(&[[1.0 / 100.0, 2.0 / 100.0]])));

    network.train(&mut input_data, expected_result.clone(), Objective::SumSquaredError);

    println!("After training : {:?}", network.feed_forward(&arr2(&[[1.0 / 100.0, 2.0 / 100.0]]))); // Should equals 0.03
    println!("After training : {:?}", network.feed_forward(&arr2(&[[2.0 / 100.0, 2.0 / 100.0]]))); // Should equals 0.04
    println!("After training : {:?}", network.feed_forward(&arr2(&[[20.0 / 100.0, 2.0 / 100.0]]))); // Should equals 0.22
}