use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

pub struct Ctrnn {
    pub y: Matrix<f64>,
    // membrane potential
    pub tau: Matrix<f64>,
    pub wji: Matrix<f64>,
    // bias
    pub theta: Matrix<f64>,
    pub i: Matrix<f64>,
    pub num_inputs: usize,
    pub nr_output_neurons: usize,
}

impl Ctrnn {
    pub fn new(num_neurons: usize, num_inputs: usize, nr_output_neurons: usize) -> Ctrnn {
        Ctrnn {
            y: Matrix::zeros(num_neurons, 1),
            tau: Matrix::new(num_neurons, 1, vec![1.0; num_neurons]),
            wji: Matrix::zeros(num_neurons, num_neurons),
            theta: Matrix::zeros(num_neurons, 1),
            i: Matrix::zeros(num_neurons, 1),
            num_inputs,
            nr_output_neurons,
        }
    }

    pub fn update(&mut self, dt: f64, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.num_inputs);
        self.i.mut_data()[0..self.num_inputs].copy_from_slice(inputs);
        let current_weights = (&self.y + &self.theta).apply(&Ctrnn::sigmoid);
        self.y = &self.y
            + ((&self.wji * current_weights) - &self.y + &self.i)
                .elediv(&self.tau)
                .apply(&|j_value| dt * j_value);
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        self.y
            .data()
            .iter()
            .skip(self.num_inputs + 1)
            .take(self.nr_output_neurons)
            .cloned()
            .map(&Ctrnn::sigmoid)
            .collect()
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}
