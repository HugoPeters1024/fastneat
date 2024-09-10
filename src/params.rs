#[derive(Clone)]
pub struct Settings {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub population_size: usize,
    pub target_species: usize,
    pub parameters: Parameters,
}

#[derive(Clone)]
pub struct Parameters {
    pub mutate_genome_add_connection: f64,
    pub mutate_genome_add_neuron: f64,
    pub mutate_genome_weight_change: f64,
    pub mutate_gene_weight_change: f64,
    pub mutate_gene_nudge_factor: f64,
    pub mutate_gene_toggle_expression: f64,
    pub compat_mismatch_genes_factor: f64,
    pub compat_mismatch_weight_factor: f64,
    pub specie_dropoff_age: usize,
    pub specie_greediness_exponent: f64,
    pub enable_elitism: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            mutate_genome_add_connection: 0.1,
            mutate_genome_add_neuron: 0.04,
            mutate_genome_weight_change: 0.8,
            mutate_gene_weight_change: 0.9,
            mutate_gene_nudge_factor: 2.0,
            mutate_gene_toggle_expression: 0.05,
            compat_mismatch_genes_factor: 2.0,
            compat_mismatch_weight_factor: 0.1,
            specie_dropoff_age: 35,
            specie_greediness_exponent: 1.5,
            enable_elitism: true,
        }
    }
}
