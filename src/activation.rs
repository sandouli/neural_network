use ndarray::Array2;



pub enum Activation {
    Identity,
    Binary(f64),
    Sigmoid,
    TanH,
    ReLU,
    LeakyReLU(f64),
    Softmax,
    LogSoftmax,
}


impl Activation {
    pub fn compute(&self, array: &Array2<f64>) -> Array2<f64> {
        match *self {
            Activation::Identity => {
                array.clone()
            },
            Activation::Binary(threshold) => {
                array.map(|v| if *v < threshold { 0.0 } else { 1.0 })
            },
            Activation::Sigmoid => {
                array.map(|v| 1.0 / (1.0 + (-v).exp()))
            },
            Activation::TanH => {
                array.map(|v| v.tanh())
            },
            Activation::ReLU => {
                array.map(|v| v.max(0.0))
            },
            Activation::LeakyReLU(slope) => {
                array.map(|v| if *v < 0.0 { v * slope } else { *v })
            },
            Activation::Softmax => {
                let mut result = array.clone();
                let mut inter = array.clone();
                for i in 0..result.rows() {
                    let max_value = array.slice(s![i, ..]).iter().cloned().fold(0./0., f64::max);
                    inter.slice_mut(s![i, ..]).assign(&array.slice(s![i, ..]).map(|v| (v - max_value).exp()));
                    let sum = inter.slice(s![i, ..]).scalar_sum();
                    result.slice_mut(s![i, ..]).assign(&inter.slice(s![i, ..]).map(|v| v / sum));
                }
                result
            },
            Activation::LogSoftmax => {

                let mut result = array.clone();

                // First implementation
                for i in 0..array.rows() {
                    let log_sum_exp = (array.slice(s![i, ..]).map(|v| v.exp()).scalar_sum() as f64).ln();
                    result.slice_mut(s![i, ..]).assign(&array.slice(s![i, ..]).map(|v| v - log_sum_exp));
                }

                // Second implementation
//                for i in 0..result.rows() {
//                    let max_value = array.slice(s![i, ..]).iter().cloned().fold(0. / 0., f64::max);
//                    let log_sum_exp = array.slice(s![i, ..]).map(|v| (v - max_value).exp()).scalar_sum() + max_value;
//                    result.slice_mut(s![i, ..]).assign(&array.slice(s![i, ..]).map(|v| v - log_sum_exp));
//                }

                // TODO : see https://github.com/deeplearning4j/nd4j/issues/2822 for other implementations ?




                result
            },
        }
    }

    pub fn compute_derivative(&self, array: &Array2<f64>) -> Array2<f64> {
        match *self {
            Activation::Identity => {
                array.map(|_| 1.0)
            },
            Activation::Binary(threshold) => {
                array.map(|v| 0.0)
            },
            Activation::Sigmoid => {
                self.compute(array).map(|v| v * (1.0 - v))
            },
            Activation::TanH => {
                array.map(|v| 1.0 - v.tanh().powi(2))
            },
            Activation::ReLU => {
                array.map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
            },
            Activation::LeakyReLU(slope) => {
                array.map(|v| if *v > 0.0 { 1.0 } else { slope })
            },
            Activation::Softmax => {
                // Testing jacobian derivative
                assert_eq!(array.rows(), 1);
                let mut result = Array2::<f64>::zeros((array.cols(), array.cols()));
                for i in 0..array.cols() {
                    for j in 0..array.cols() {
                        let kronecker_delta = if i == j {
                            1.0
                        } else {
                            0.0
                        };
                        result[[i, j ]] = array[[0, i]] * (kronecker_delta - array[[0, j]]);
                    }
                }
                result
            },
            Activation::LogSoftmax => {
                Activation::Softmax.compute_derivative(array)
                // TODO : implement
            }
        }
    }
}


#[cfg(test)]
mod tests {

    // TODO : replace assert_eq by assert_ulps_eq because floating point arithmetic is not quite precise and need a range

    use ndarray::arr2;
    use super::*;

    fn test_activation_function(activation_function: Activation, input: Array2<f64>, expected_result: Array2<f64>, expected_derivative: Array2<f64>) {

        assert_eq!(expected_result, activation_function.compute(&input));

        assert_eq!(expected_derivative, activation_function.compute_derivative(&input));    // TODO : Implement all derivatives
    }

    #[test]
    fn identity() {
        let input = arr2(&[[1., 2., 3., 4.], [5., 6., 7., 8.]]);

        test_activation_function(
            Activation::Identity,
            input.clone(),
            input,
            arr2(&[
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
            ]),
        );
    }

    #[test]
    fn binary() {
        test_activation_function(
            Activation::Binary(0.5),
            arr2(&[
                [0.1, 0.7, 0.9, 0.2],
                [0.6, 0.5, 0.4, 0.49],
            ]),
            arr2(&[
                [0., 1., 1., 0.],
                [1., 1., 0., 0.],
            ]),
            arr2(&[
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ]),
        );
    }

    #[test]
    fn sigmoid() {
        test_activation_function(
            Activation::Sigmoid,
            arr2(&[
                [-5., -1., 0., 6.],
                [1., 0.1, 0.11, 11.],
            ]),
            arr2(&[
                [
                    0.0066928509242848554,
                    0.2689414213699951,
                    0.5,
                    0.9975273768433653,
                ],
                [
                    0.7310585786300049,
                    0.52497918747894,
                    0.5274723043445937,
                    0.999983298578152,
                ],
            ]),
            arr2(&[
                [
                    0.006648056670790155,
                    0.19661193324148185,
                    0.25,
                    0.002466509291359931,
                ],
                [
                    0.19661193324148185,
                    0.24937604019289197,
                    0.24924527249399803,
                    0.00001670114291046157,
                ],
            ]),
        );
    }

    #[test]
    fn tanh() {
        test_activation_function(
            Activation::TanH,
            arr2(&[
                [-5., -1., 0., 6.],
                [1., 0.1, 0.11, 11.],
            ]),
            arr2(&[
                [
                    -0.9999092042625951,
                    -0.7615941559557649,
                    0.,
                    0.9999877116507956,
                ],
                [
                    0.7615941559557649,
                    0.09966799462495582,
                    0.10955847021442953,
                    0.9999999994421064,
                ],
            ]),
            arr2(&[
                [
                    0.0001815832309438603,
                    0.41997434161402614,
                    1.0,
                    0.000024576547405286142,
                ],
                [
                    0.41997434161402614,
                    0.9900662908474398,
                    0.987996941604274,
                    0.000000001115787240379973,
                ],
            ]),
        );
    }

    #[test]
    fn relu() {
        test_activation_function(
            Activation::ReLU,
            arr2(&[
                [-5., -1., 0., -0.1],
                [1., 0.1, 0.01, 11.],
            ]),
            arr2(&[
                [0., 0., 0., 0.],
                [1., 0.1, 0.01, 11.],
            ]),
            arr2(&[
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
            ]),
        );
    }

    #[test]
    fn leaky_relu() {
        test_activation_function(
            Activation::LeakyReLU(0.3),
            arr2(&[
                [-5., -1., 0., -0.1],
                [1., 0.1, 0.01, 11.],
            ]),
            arr2(&[
                [-1.5, -0.3, 0., -0.03],
                [1., 0.1, 0.01, 11.],
            ]),
            arr2(&[
                [0.3, 0.3, 0.3, 0.3],
                [1., 1., 1., 1.],
            ]),
        );
    }

    #[test]
    fn softmax() {
        test_activation_function(
            Activation::Softmax,
            arr2(&[
                [-5., -1., 0., -0.1],
                [1., 0.1, 0.01, 11.],
            ]),
            arr2(&[
                [
                    0.002955946738114491,
                    0.16138922349755833,
                    0.43870139354252835,
                    0.3969534362217987,
                ],
                [
                    0.00004539626502553861,
                    0.000018456744024927285,
                    0.00001686819394294945,
                    0.9999192787970065,
                ],
            ]),
            arr2(&[
                [-1.5, -0.5, -0.25, -0.275],
                [0.0, -0.225, -0.2475, 2.5],
            ]),
        );
    }



}