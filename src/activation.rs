use ndarray::Array2;



pub enum Activation {
    Identity,
    Binary(f64),
    Sigmoid,
    TanH,
    ReLU,
    LeakyReLU(f64),
    Softmax,
}


impl Activation {
    pub fn compute(&self, array: Array2<f64>) -> Array2<f64> {
        match *self {
            Activation::Identity => {
                array
            },
            Activation::Binary(threshold) => {
                array.map(|v| if *v < threshold { 0.0 } else { 1.0 })
            }
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
            }
            Activation::Softmax => {
                let sum_exp: f64 = array.map(|v| v.exp()).iter().sum();
                array.map(|v| v.exp() / sum_exp)
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use super::*;

    fn test_activation_function(activation_function: Activation, input: Array2<f64>, expected_result: Array2<f64>) {

        assert_eq!(expected_result, activation_function.compute(input));

        // TODO : derivative results too
    }

    #[test]
    fn identity() {
        let input = arr2(&[[1., 2., 3., 4.], [5., 6., 7., 8.]]);

        test_activation_function(Activation::Identity, input.clone(), input);
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
                    0.00000011252180729828334,
                    0.000006143482516872218,
                    0.00001669971688906959,
                    0.000015110528711837235,
                ],
                [
                    0.00004539453695996848,
                    0.00001845604144589644,
                    0.00001686755183406304,
                    0.999881215619835,
                ],
            ]),
        );

        // Sum of softmax function output should be equal to one (1.0)
        assert_eq!(1.0, Activation::Softmax.compute(arr2(&[[-5., -1., 0., -0.1], [1., 0.1, 0.01, 11.]])).iter().sum());
    }



}