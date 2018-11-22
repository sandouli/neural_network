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
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use builder::NeuralNetworkBuilder;
use activation::Activation;




fn main() {
    println!("Hello world!");

    let input_data = Array2::<f64>::ones((1, 2));

    let mut network = NeuralNetworkBuilder::new(input_data.cols())
        .layer(4, Activation::TanH)
        .layer(4, Activation::TanH)
        .layer(1, Activation::TanH)
        .build();



    let result = network.feed_forward(input_data);
    println!("{:?}", result);


    //tests_array();
}

fn tests_array() {

    let mut input = Array2::<f64>::random((3, 3), Range::new(0.0, 2.0));
    input.slice_mut(s![.., -1]).fill(1.0);

    let mut output = Array2::<f64>::ones((input.rows(), 3));

    let mut weights = Array2::<f64>::random((input.cols(), output.cols()), Range::new(0.0, 1.0));
    weights.slice_mut(s![.., -1]).fill(0.1);

    output = input.dot(&weights);

    println!("{:?}", output);
    let mut test = Array2::<f64>::ones((input.rows(), output.cols() + 1));
    test.slice_mut(s![.., ..-1]).assign(&input.dot(&weights));
    println!("{:?}", test);
    println!("{:?}", test.map(|x| x.tanh()));
    println!("{}", -0.17f64.tanh());
    println!("{}", 0.253f64.tanh());
//    test.slice_mut(s![.., -1]).fill(1.0);
//    println!("{:?}", test);

}
