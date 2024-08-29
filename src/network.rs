use crate::{activation::Activation, matrix::Matrix};

pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activation: Activation,
    data: Vec<Matrix>,
    learning_rate: f64,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let len = layers.len();
        let mut weights = Vec::with_capacity(len - 1);
        let mut biases = Vec::with_capacity(len - 1);

        for i in 0..len - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network { layers, weights, biases, activation, data: vec![], learning_rate }
    }

    fn feed_forward(&mut self, input: Matrix) -> Matrix {
        assert_eq!(input.rows, self.layers[0]);
        let mut current = input;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i].dot(&current).add(&self.biases[i]).map(self.activation.func);
            self.data.push(current.clone());
        }

        current
    }

    fn back_propagate(&mut self, input: Matrix, target: Matrix) {
        assert_eq!(target.rows, self.layers[self.layers.len() - 1]);

        let mut errors = target.sub(&input);
        let mut gradients = input.clone().map(self.activation.derivate); //.hadamard(&errors);

        /* for i in (0..self.layers.len() - 1).rev() {
            let deltas = errors.hadamard(&gradients).mul(learning_rate);
            self.biases[i] = self.biases[i].add(&deltas);
            self.weights[i] = deltas.dot(&self.data[i].transpose()).add(&self.weights[i]);

            errors = self.weights[i].transpose().dot(&errors);
            gradients = self.data[i].map(self.activation.derivate);
        } */

       for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.hadamard(&errors).mul(self.learning_rate);
            self.biases[i] = self.biases[i].add(&gradients);
            self.weights[i] = gradients.dot(&self.data[i].transpose()).add(&self.weights[i]);
            errors = self.weights[i].transpose().dot(&errors);
            gradients = self.data[i].map(self.activation.derivate);
        }
    }

    pub fn predict(&mut self, input: Matrix) -> Matrix {
        self.feed_forward(input)
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        assert_eq!(inputs.len(), targets.len());
        for i in 1..=epochs {
            if i % 10_000 == 0 {
                println!("Epoch: {}", i);
                println!("Loss: {}\n", self.feed_forward(Matrix::from_vec(inputs[0].clone())).sub(&Matrix::from_vec(targets[0].clone())).sum());
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from_vec(inputs[j].clone()));
                self.back_propagate(outputs,Matrix::from_vec( targets[j].clone()));
            }   
        }
    }
}