use rand::Rng;
use std::collections::{BTreeMap, HashMap};

use crate::params::*;
use crate::population::full_sorted_outer_join;

#[derive(Clone, Debug)]
pub struct Gene {
    pub innovation_number: usize,
    pub neuron_from: usize,
    pub neuron_to: usize,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Clone, Debug)]
pub struct Neuron {
    pub tau: f64,
    pub is_bias: bool,
}

#[derive(Debug, Clone)]
pub struct Genome {
    // innovation number -> Gene
    pub genes: BTreeMap<usize, Gene>,
    all_innovation_numbers: Vec<usize>,
    pub neurons: HashMap<usize, Neuron>,
    pub fitness: f64,
    max_neuron_id: usize,
    pub specie_idx: Option<usize>,
}

impl Genome {
    pub fn empty(settings: &Settings) -> Genome {
        let mut all_neurons = HashMap::new();
        let mut max_neuron_id = 0;
        for i in 0..settings.num_inputs + settings.num_outputs {
            all_neurons.insert(
                i,
                Neuron {
                    tau: 0.1,
                    is_bias: false,
                },
            );
            max_neuron_id = max_neuron_id.max(i);
        }
        return Genome {
            genes: BTreeMap::new(),
            all_innovation_numbers: Vec::new(),
            neurons: all_neurons,
            max_neuron_id,
            fitness: 0.0,
            specie_idx: None,
        };
    }

    pub fn add_neuron(&mut self, id: usize, neuron: Neuron) {
        self.neurons.insert(id, neuron);
    }

    pub fn add_gene(&mut self, gene: Gene) {
        // TODO: make this configurable
        if gene.neuron_from == gene.neuron_to {
            return;
        }
        if let Some(existing_gene) = self.genes.get_mut(&gene.innovation_number) {
            existing_gene.enabled = gene.enabled;
            existing_gene.weight = gene.weight;
            return;
        }

        if gene.neuron_from > self.max_neuron_id {
            self.max_neuron_id = gene.neuron_from;
        }

        if gene.neuron_to > self.max_neuron_id {
            self.max_neuron_id = gene.neuron_to;
        }

        if !self.neurons.contains_key(&gene.neuron_from) {
            self.neurons.insert(
                gene.neuron_from,
                Neuron {
                    tau: 0.1,
                    is_bias: false,
                },
            );
        }
        if !self.neurons.contains_key(&gene.neuron_to) {
            self.neurons.insert(
                gene.neuron_to,
                Neuron {
                    tau: 0.1,
                    is_bias: false,
                },
            );
        }
        self.all_innovation_numbers.push(gene.innovation_number);
        self.genes.insert(gene.innovation_number, gene);
    }

    pub fn get_num_neurons(&self) -> usize {
        return self.max_neuron_id + 1;
    }

    pub fn sample_gene_mut(&mut self) -> &mut Gene {
        let mut rng = rand::thread_rng();
        let gene_idx = rng.gen_range(0..self.all_innovation_numbers.len());
        let innov_number = self.all_innovation_numbers[gene_idx];
        self.genes.get_mut(&innov_number).unwrap()
    }

    pub fn sample_neuron_id(&self) -> usize {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.neurons.len());
        self.neurons
            .keys()
            .skip(idx)
            .take(1)
            .next()
            .unwrap()
            .clone()
    }

    pub fn compatibility(&self, other: &Genome, genes_factor: f64, weight_factor: f64) -> f64 {
        let mut num_mismatch = 0.0;
        let mut weight_diff_sum = 0.0;
        for (l, r) in full_sorted_outer_join(self.genes.values(), other.genes.values(), |a, b| {
            a.innovation_number.cmp(&b.innovation_number)
        }) {
            match (l, r) {
                (None, _) => num_mismatch += 1.0,
                (_, None) => num_mismatch += 1.0,
                (Some(lhs), Some(rhs)) => {
                    if !lhs.enabled || !rhs.enabled {
                        num_mismatch += 1.0;
                        continue;
                    }

                    weight_diff_sum += (lhs.weight - rhs.weight).abs();
                }
            }
        }

        let n = self.genes.len().max(other.genes.len()).max(1) as f64;
        return genes_factor * (num_mismatch / n) + weight_factor * weight_diff_sum;
    }

    pub fn print_dot(&self) {
        for (neuron_idx, neuron) in self.neurons.iter() {
            println!(
                "n{} [label=\"n{} {:.2}\"]",
                neuron_idx, neuron_idx, neuron.tau
            )
        }
        for gene in self.genes.values() {
            if gene.enabled {
                println!(
                    "n{} -> n{} [label=\"{:.2}\"]",
                    gene.neuron_from, gene.neuron_to, gene.weight
                );
            }
        }
    }
}
