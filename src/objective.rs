// Loss/cost functions

use ndarray::{Array2, Zip};

#[derive(Copy, Clone)]
pub enum Objective {
    // Classification : predicts a label
    Log,    // Related to Cross Entropy ?
    Focal,
    Exponential,
    Hinge,
    CrossEntropy,

    // Regression : predicts a quantity
    SumSquaredError,    // Currently testing this one
    MeanSquaredError,
    MeanAbsoluteError,
    Huber,
    LogCosh,
    Quantile,

    // ?????
    Likelihood,
}

impl Objective {
    pub fn calculate_error(&self, output: &Array2<f64>, expected_output: &Array2<f64>) -> f64 {
        assert_eq!(output.rows(), expected_output.rows());
        assert_eq!(output.cols(), expected_output.cols());
        match *self {
            Objective::SumSquaredError => {
                let mut squared_diffs: Array2<f64> = Array2::zeros(expected_output.dim());
                Zip::from(&mut squared_diffs)
                    .and(output.view())
                    .and(expected_output.view())
                    .apply(|d, expected, approx| *d = (expected - approx).powi(2));


                let mut error = Array2::<f64>::zeros((output.rows(), 1));
                for i in 0..error.rows() {
                    error[[i, 0]] = 0.5 * squared_diffs.slice(s![i, ..]).scalar_sum();
                }

                error.scalar_sum() / error.rows() as f64
            },
            Objective::CrossEntropy => {
                // Trying https://www.youtube.com/watch?v=PHP8beSz5o4
                let mut result = 0.0;
                // Epsilon is used to prevent NaN when trying ln(0.0)
                let epsilon = 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001;
                for i in 0..output.rows() {
                    for j in 0..output.cols() {
                        //println!("Expected result : {:?}", expected_output[[i, j]]);
                        //println!("Calculated result : {:?}", output[[i, j]]);
                        //println!("Calculated result ln : {:?}", output[[i, j]].ln());
                        //println!("Formula : {:?}", expected_output[[i, j]] * output[[i, j]].ln());

                        result += expected_output[[i, j]] * (output[[i, j]] + epsilon).ln();
                        //println!("Temp result : {:?}", result);
                    }
                }

                (-1.0 / output.rows() as f64) * result
            }
            _ => unreachable!(),
        }
    }

    pub fn compute_derivative(&self, output: &Array2<f64>, expected_output: &Array2<f64>) -> Array2<f64> {
        assert_eq!(output.rows(), expected_output.rows());
        assert_eq!(output.cols(), expected_output.cols());

        match *self {
            Objective::SumSquaredError => {
                2.0 * (expected_output - output)    // TODO : probably not good, look it up
            },
            Objective::CrossEntropy => {
                // Trying https://www.youtube.com/watch?v=PHP8beSz5o4
                //output - expected_output
                expected_output - output
            },
            _ => unreachable!(),
        }

    }
}


#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use super::*;

    #[test]
    fn identity() {
        let objective_function = Objective::SumSquaredError;

        let output = arr2(
            &[
                [0.45],
                [0.62],
            ]);
        let expected_output = arr2(
            &[
                [0.],
                [0.22],
            ]);

        let error = arr2(
            &[
                [0.10125],
                [0.08000000000000002],

            ]
        );

        assert_eq!(objective_function.calculate_error(&output, &expected_output), error.scalar_sum() as f64 / error.rows() as f64);
    }

}