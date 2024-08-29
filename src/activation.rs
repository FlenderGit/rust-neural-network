use std::f64::consts::E;

pub struct Activation {
    pub func: fn(f64) -> f64,
    pub derivate: fn(f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    func: |x| 1.0 / (1.0 + E.powf(-x)),
    derivate: |x| x * (1.0 - x),
};