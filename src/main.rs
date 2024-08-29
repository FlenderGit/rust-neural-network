use neural_network::{activation::SIGMOID, matrix::Matrix, network::Network};

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new(vec![2, 3, 1], SIGMOID, 0.5);

    network.train(inputs, targets, 100000);

    println!("{:?}", network.predict(Matrix::from_vec(vec![0.0, 0.0])));
    println!("{:?}", network.predict(Matrix::from_vec(vec![0.0, 1.0])));
    println!("{:?}", network.predict(Matrix::from_vec(vec![1.0, 0.0])));
    println!("{:?}", network.predict(Matrix::from_vec(vec![1.0, 1.0])));
}
