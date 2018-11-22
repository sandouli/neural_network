// Loss/cost functions

pub enum Objective {
    // Classification : predicts a label
    Log,    // Related to Cross Entropy ?
    Focal,
    Exponential,
    Hinge,

    // Regression : predicts a quantity
    MeanSquaredError,
    MeanAbsoluteError,
    Huber,
    LogCosh,
    Quantile,

    // ?????
    Likelihood,
}