pub const MUTATE_GENOME_ADD_CONNECTION: f64 = 0.08;
pub const MUTATE_GENOME_ADD_NEURON: f64 = 0.04;
pub const MUTATE_GENOME_WEIGHT_CHANGE: f64 = 0.8;
pub const MUTATE_GENE_WEIGHT_CHANGE: f64 = 2.5;
pub const MUTATE_GENE_TOGGLE_EXPRESSION: f64 = 0.05;

pub const COMPAT_MISMATCH_GENES_FACTOR: f64 = 2.0;
pub const COMPAT_MISMATCH_WEIGHT_FACTOR: f64 = 1.0;

pub const SPECIE_DROPOFF_AGE: usize = 35;
pub const SPECIE_GREEDINESS_EXPONENT: f64 = 2.0;

pub const TOPOLOGY_ALLOW_SELF_CONNECTIONS: bool = false;

#[derive(Clone)]
pub struct Settings {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub population_size: usize,
    pub target_species: usize,
}
