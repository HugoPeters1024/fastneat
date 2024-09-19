use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

use crate::params::ActivationFunction;

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
    pub activation_function: ActivationFunction,
}

impl Ctrnn {
    pub fn new(
        num_neurons: usize,
        num_inputs: usize,
        nr_output_neurons: usize,
        activation_function: ActivationFunction,
    ) -> Ctrnn {
        Ctrnn {
            y: Matrix::zeros(num_neurons, 1),
            tau: Matrix::new(num_neurons, 1, vec![1.0; num_neurons]),
            wji: Matrix::zeros(num_neurons, num_neurons),
            theta: Matrix::zeros(num_neurons, 1),
            i: Matrix::zeros(num_neurons, 1),
            num_inputs,
            nr_output_neurons,
            activation_function,
        }
    }

    pub fn update(&mut self, dt: f64, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.num_inputs);
        self.i.mut_data()[0..self.num_inputs].copy_from_slice(inputs);
        let current_weights = (&self.y + &self.theta).apply(&|x| self.activation(x));
        self.y = &self.y
            + ((&self.wji * current_weights) - &self.y + &self.i)
                .elediv(&self.tau)
                .apply(&|j_value| dt * j_value);
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        self.y
            .data()
            .iter()
            .skip(self.num_inputs)
            .take(self.nr_output_neurons)
            .cloned()
            .map(|x| self.activation(x))
            .collect()
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
        }
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}
