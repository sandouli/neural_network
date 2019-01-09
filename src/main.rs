extern crate rand;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;

extern crate mnist;
extern crate image;




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

    test_mnist();

    //test_simple_softmax();

//    test_addition();

//    test_equal();

    // https://news.ycombinator.com/item?id=18145622
    //test_basic();
}

fn test_basic() {
    let training_size = 4;

    let epoch = 1500;

    println!("Creating neural network");
    let mut network = NeuralNetworkBuilder::new(3)
        .layer(4, Activation::Sigmoid)
        .layer(1, Activation::Sigmoid)
        .build();

    println!("Preparing training set");
    let mut training_input_data = Array2::<f64>::zeros((training_size, 3));
    let mut training_expected_result = Array2::<f64>::zeros((training_size, 1));

    for i in 0..training_size {
        if i > (training_size / 2) - 1  {
            training_input_data[[i, 0]] = 1.0;
        }
        if i%2 == 1 {
            training_input_data[[i, 1]] = 1.0;
        }
        training_input_data[[i, 2]] = 1.0;

        if i != 0 && i != 3 {
            training_expected_result[[i, 0]] = 1.0;
        }
    }

    println!("{}", network.feed_forward(&training_input_data));

    println!("Starting training");
    for _ in 0..epoch {
        network.train(
            &mut training_input_data,
            training_expected_result.clone(),
            Objective::SumSquaredError,
            1,
            1.0
        );
    }

    println!("{}", network.feed_forward(&training_input_data));
}

fn test_equal() {
    let training_size = 100000;
    let test_size = 100;
    let batch_size = 1000;
    let epoch = 8;
    let learning_rate = 0.01;
    let denominator = 1000000.0;

    println!("Creating neural network");
    let mut network = NeuralNetworkBuilder::new(1)
        //.layer(10, Activation::LeakyReLU(0.3))
        //.layer(10, Activation::LeakyReLU(0.3))
        .layer(10, Activation::ReLU)
        .layer(1, Activation::Identity)
        .build();

    println!("Preparing training set");
    let mut training_input_data = Array2::<f64>::zeros((training_size, 1));
    let mut training_expected_result = Array2::<f64>::zeros((training_size, 1));

    for i in 0..training_size {
        training_input_data[[i, 0]] = i as f64 / denominator;

        training_expected_result[[i, 0]] = i as f64 / denominator;
    }

    println!("Preparing test set");
    let mut test_input_data = Array2::<f64>::zeros((test_size, 1));
    let mut test_expected_result = Array2::<f64>::zeros((test_size, 1));

    for i in 0..test_size {
        test_input_data[[i, 0]] = i as f64 + 1000.0;

        test_expected_result[[i, 0]] = i as f64 + 1000.0;
    }


    println!("Starting training");
    for _ in 0..epoch {
        network.train(
            &mut training_input_data,
            training_expected_result.clone(),
            Objective::SumSquaredError,
//            Objective::CrossEntropy,
            batch_size,
            learning_rate
        );
    }


    let result_test = network.feed_forward(&test_input_data);

    let mut good_guesses = 0;

    for row in 0..result_test.rows() {
        println!("(Result, Expected) => ({}, {})", result_test[[row, 0]], test_expected_result[[row, 0]]);
    }
}

fn test_addition() {
    let training_size = 100000;
    let test_size = 100;

    println!("Preparing training set");
    let mut training_input_data = Array2::<f64>::zeros((training_size, 2));
    let mut training_expected_result = Array2::<f64>::zeros((training_size, 1));

    let denominator = 1000000.0;

    for i in 0..training_size {
        training_input_data[[i, 0]] = i as f64 / denominator;
        training_input_data[[i, 1]] = (i + i) as f64 / denominator;

        training_expected_result[[i, 0]] = training_input_data[[i, 0]] + training_input_data[[i, 1]];
    }


    println!("Preparing test set");
    let mut test_input_data = Array2::<f64>::zeros((test_size, 2));
    let mut test_expected_result = Array2::<f64>::zeros((test_size, 1));

    for i in 0..test_size {
        test_input_data[[i, 0]] = i as f64;
        test_input_data[[i, 1]] = i as f64 + 1.0;

        test_expected_result[[i, 0]] = test_input_data[[i, 0]] + test_input_data[[i, 1]];
    }

    let mut network = NeuralNetworkBuilder::new(2)
        .layer(1, Activation::Identity)
        .build();

    println!("Starting training");

    for _ in 0..1 {
        network.train(
            &mut training_input_data,
            training_expected_result.clone(),
            Objective::SumSquaredError,
//            Objective::CrossEntropy,
            10,
            0.001
        );
    }


    let result_test = network.feed_forward(&test_input_data);

    let mut good_guesses = 0;

    for row in 0..result_test.rows() {
        println!("(Result, Expected) => ({}, {})", result_test[[row, 0]], test_expected_result[[row, 0]]);

    }
    println!("Number of good guesses : {:?}", good_guesses);
    println!("Number of result rows : {:?}", result_test.rows());
    println!("{}% of good guesses", (good_guesses as f64 / result_test.rows() as f64) * 100.0);
}


fn test_mnist() {
    use mnist::{Mnist, MnistBuilder};

    let training_set_size = 1; // 50000
    let test_set_size: usize = 1; // 10000
    let epoch = 10;
    let batch_size = 32;
    let learning_rate = 0.1;
    let objective_function = Objective::CrossEntropy;


    println!("Creating neural network");
    let mut network = NeuralNetworkBuilder::new(28 * 28)
        .layer(512, Activation::ReLU)
        .layer(10, Activation::Softmax)
        .build();

    println!("Building MNIST data");
    let Mnist { trn_img, trn_lbl, val_img, val_lbl, tst_img, tst_lbl } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_set_size as u32)
        .validation_set_length(10_000)
        .test_set_length(test_set_size as u32)
        .finalize();



    println!("Preparing training set");
    let mut training_input_data = Array2::<f64>::zeros((training_set_size, 28 * 28));
    let mut training_expected_result = Array2::<f64>::zeros((training_set_size, 10));

    for i in 0..training_set_size {
        for j in 0..(28*28) {
            training_input_data[[i, j]] = trn_img[i * 784 + j] as f64;
        }
        training_expected_result[[i, trn_lbl[i] as usize]] = 1.0;
    }

    println!("Normalizing training set");
    training_input_data /= 255.0;


    println!("Preparing test set");
    let mut test_input_data = Array2::<f64>::zeros((test_set_size, 28 * 28));
    let mut test_expected_result = Array2::<f64>::zeros((test_set_size, 10));

    for i in 0..test_set_size {
        for j in 0..(28*28) {
            test_input_data[[i, j]] = tst_img[i * 784 + j] as f64;
        }
        test_expected_result[[i, tst_lbl[i] as usize]] = 1.0;
    }

    println!("Normalizing test set");
    test_input_data /= 255.0;





    println!("Starting training");

    for i in 0..epoch {
        println!("Epoch {}", i);
        network.train(
            &mut training_input_data,
            training_expected_result.clone(),
            objective_function,
            batch_size,
            learning_rate
        );
    }



    let result_test = network.feed_forward(&test_input_data);

    let mut good_guesses = 0;

    for row in 0..result_test.rows() {
        let mut my_index = 0;
        for i in 0..result_test.cols() {
            if result_test[[row as usize, i]] > result_test[[row as usize, my_index]] {
                my_index = i;
            }
        }
        if my_index == tst_lbl[row as usize] as usize {
            good_guesses += 1;
        }
    }
    println!("Number of good guesses : {:?}", good_guesses);
    println!("Number of result rows : {:?}", result_test.rows());
    println!("{}% of good guesses", (good_guesses as f64 / result_test.rows() as f64) * 100.0);








/*
    println!("Expected input for zero : ");
    let expected_input = network.get_expected_input(&arr2(&[[0., 0., 0., 3., 0., 0., 0., 0., 0., 0.]]));
    println!("{:?}", 255.0 * &expected_input);


    let mut my_vec: Vec<f64> = expected_input.row(0).to_vec();

    let min_value = my_vec.iter().cloned().fold(0./0., f64::min);
    let max_value = my_vec.iter().cloned().fold(0./0., f64::max);
    for i in 0..my_vec.len() {
        my_vec[i] = (my_vec[i] + min_value) / (min_value + max_value) * 255.0;
    }

    let mut pixels: [u8; 784] = [0; 28 * 28];

    for i in 0..my_vec.len() {
        pixels[i] = my_vec[i] as u8;
    }





    use image::ColorType;
    use image::png::PNGEncoder;
    use std::fs::File;

    let output = File::create("zero.png").unwrap();
    let encoder = PNGEncoder::new(output);
    encoder.encode(&pixels, 28, 28, ColorType::Gray(8));

*/


    create_image_heat_map("zero", &network.get_expected_input(&arr2(&[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])));
    create_image_heat_map("one", &network.get_expected_input(&arr2(&[[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])));
    create_image_heat_map("two", &network.get_expected_input(&arr2(&[[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])));
    create_image_heat_map("three", &network.get_expected_input(&arr2(&[[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])));
    create_image_heat_map("four", &network.get_expected_input(&arr2(&[[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])));
    create_image_heat_map("five", &network.get_expected_input(&arr2(&[[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])));
    create_image_heat_map("six", &network.get_expected_input(&arr2(&[[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])));
    create_image_heat_map("seven", &network.get_expected_input(&arr2(&[[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])));
    create_image_heat_map("eight", &network.get_expected_input(&arr2(&[[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])));
    create_image_heat_map("nine", &network.get_expected_input(&arr2(&[[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])));

}

fn test_simple_softmax() {
    let training_set_size = 2;
    let test_set_size: usize = 2;
    let epoch = 10000;
    let batch_size = 2;
    let learning_rate = 0.1;
    let objective_function = Objective::CrossEntropy;


    println!("Creating neural network");
    let mut network = NeuralNetworkBuilder::new(2)
        .layer(2, Activation::ReLU)
        .layer(2, Activation::Softmax)
        .build();



    println!("Preparing training set");
    let mut training_input_data = Array2::<f64>::zeros((training_set_size, 2));
    let mut training_expected_result = Array2::<f64>::zeros((training_set_size, 2));

    {
        training_input_data[[0, 0]] = 1.0;
        training_input_data[[0, 1]] = 0.0;
        training_expected_result[[0, 0]] = 1.0;
        training_expected_result[[0, 1]] = 0.0;

        training_input_data[[1, 0]] = 0.0;
        training_input_data[[1, 1]] = 1.0;
        training_expected_result[[1, 0]] = 0.0;
        training_expected_result[[1, 1]] = 1.0;
    }


    println!("Preparing test set");
    let mut test_input_data = Array2::<f64>::zeros((test_set_size, 2));
    let mut test_expected_result = Array2::<f64>::zeros((test_set_size, 2));

    {
        test_input_data[[0, 0]] = 1.0;
        test_input_data[[0, 1]] = 0.0;
        test_expected_result[[0, 0]] = 1.0;
        test_expected_result[[0, 1]] = 0.0;

        test_input_data[[1, 0]] = 0.0;
        test_input_data[[1, 1]] = 1.0;
        test_expected_result[[1, 0]] = 0.0;
        test_expected_result[[1, 1]] = 1.0;
    }




    println!("Starting training");

    for _ in 0..epoch {
        network.train(
            &mut training_input_data,
            training_expected_result.clone(),
            objective_function,
            batch_size,
            learning_rate
        );
    }



    let result_test = network.feed_forward(&test_input_data);

    println!("Result test : {:?}", result_test);

}


fn create_image_heat_map(name: &str, expected_input: &Array2<f64>) {


    let mut my_vec: Vec<f64> = expected_input.row(0).to_vec();

    let min_value = my_vec.iter().cloned().fold(0./0., f64::min);
    let max_value = my_vec.iter().cloned().fold(0./0., f64::max);
    for i in 0..my_vec.len() {
        my_vec[i] = (my_vec[i] + min_value) / (min_value + max_value) * 255.0;
    }

    let mut pixels: [u8; 784] = [0; 28 * 28];

    for i in 0..my_vec.len() {
        pixels[i] = my_vec[i] as u8;
    }





    use image::ColorType;
    use image::png::PNGEncoder;
    use std::fs::File;

    let mut file_name = name.to_owned();
    file_name.push_str(".png");

    let output = File::create(file_name).unwrap();
    let encoder = PNGEncoder::new(output);
    encoder.encode(&pixels, 28, 28, ColorType::Gray(8));

    // TODO : how to have colors on heatmap ?
}