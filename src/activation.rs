use ndarray::Array2;


pub enum Activation {
    Identity,
    Binary(f64),
    Sigmoid,
    TanH,
    ReLU,
}


impl Activation {
    pub fn compute(&self, array: Array2<f64>) -> Array2<f64> {
        match *self {
            Activation::Identity => {
                array
            },
            Activation::Binary(threshold) => {
                array.map(|v| if v < &threshold { 0.0 } else { 1.0 })
            }
            Activation::Sigmoid => {
                array
            },
            Activation::TanH => {
                array.map(|v| v.tanh())
            },
            Activation::ReLU => {
                array
            },
        }
    }
}
