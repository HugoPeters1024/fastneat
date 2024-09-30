#[derive(Clone)]
pub struct Settings {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub population_size: usize,
    pub target_species: usize,
    pub parameters: Parameters,
}

#[derive(Clone)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
}

#[derive(Clone)]
pub struct Parameters {
    pub mutate_genome_add_connection: f64,
    pub mutate_genome_add_neuron: f64,
    pub mutate_genome_add_bias_neuron: f64,
    pub mutate_genome_weight_change: f64,
    pub mutate_gene_weight_change: f64,
    pub mutate_gene_nudge_factor: f64,
    pub mutate_genome_tau_change: f64,
    pub mutate_neuron_tau_change: f64,
    pub mutate_neuron_tau_nudge_factor: f64,
    pub mutate_gene_toggle_expression: f64,
    pub compat_mismatch_genes_factor: f64,
    pub compat_mismatch_weight_factor: f64,
    pub compat_mismatch_tau_factor: f64,
    pub specie_greediness: f64,
    pub specie_threshold_initial: f64,
    pub specie_threshold_nudge_factor: f64,
    pub specie_dropoff_age: usize,
    pub enable_elitism: bool,
    pub activation_function: ActivationFunction,
    pub allow_recurrent_inputs: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            mutate_genome_add_connection: 0.5,
            mutate_genome_add_neuron: 0.04,
            mutate_genome_add_bias_neuron: 0.02,
            mutate_genome_weight_change: 0.8,
            mutate_gene_weight_change: 0.9,
            mutate_gene_nudge_factor: 1.5,
            mutate_genome_tau_change: 0.3,
            mutate_neuron_tau_change: 0.5,
            mutate_neuron_tau_nudge_factor: 0.05,
            mutate_gene_toggle_expression: 0.2,
            compat_mismatch_genes_factor: 1.0,
            compat_mismatch_weight_factor: 1.0,
            compat_mismatch_tau_factor: 1.0,
            specie_greediness: 1.0,
            specie_dropoff_age: 15,
            specie_threshold_initial: 10.0,
            specie_threshold_nudge_factor: 1.8,
            enable_elitism: true,
            activation_function: ActivationFunction::Sigmoid,
            allow_recurrent_inputs: true,
        }
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
