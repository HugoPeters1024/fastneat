use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
};

use rand::Rng;
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

const MUTATE_GENOME_ADD_CONNECTION: f64 = 0.005;
const MUTATE_GENOME_ADD_NEURON: f64 = 0.005;
const MUTATE_GENOME_WEIGHT_CHANGE: f64 = 0.8;
const MUTATE_GENE_WEIGHT_CHANGE: f64 = 0.9;
const MUTATE_GENE_TOGGLE_EXPRESSION: f64 = 0.001;

pub fn full_sorted_outer_join<I, J, F>(
    iter1: I,
    iter2: J,
    mut cmp: F,
) -> Vec<(Option<I::Item>, Option<J::Item>)>
where
    I: Iterator,
    J: Iterator,
    F: FnMut(&I::Item, &J::Item) -> Ordering,
{
    let mut result = Vec::new();

    let mut iter1 = iter1.peekable();
    let mut iter2 = iter2.peekable();

    while iter1.peek().is_some() || iter2.peek().is_some() {
        match (iter1.peek(), iter2.peek()) {
            (Some(a), Some(b)) => match cmp(a, b) {
                Ordering::Less => {
                    result.push((Some(iter1.next().unwrap()), None));
                }
                Ordering::Greater => {
                    result.push((None, Some(iter2.next().unwrap())));
                }
                Ordering::Equal => {
                    result.push((Some(iter1.next().unwrap()), Some(iter2.next().unwrap())));
                }
            },
            (Some(_), None) => {
                result.push((Some(iter1.next().unwrap()), None));
            }
            (None, Some(_)) => {
                result.push((None, Some(iter2.next().unwrap())));
            }
            (None, None) => unreachable!(),
        }
    }

    result
}

struct Ctrnn {
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
            tau: Matrix::new(num_neurons, 1, vec![0.1; num_neurons]),
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
            .skip(self.num_inputs)
            .take(self.nr_output_neurons)
            .cloned()
            .collect()
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

struct Population {
    innovation_storage: HashMap<(usize, usize), usize>,
    innovation_counter: usize,
    num_inputs: usize,
    num_outputs: usize,
    members: Vec<Genome>,
}

impl Population {
    fn new(inputs: usize, outputs: usize, pop_size: usize) -> Population {
        let mut pop = Population {
            innovation_storage: HashMap::new(),
            innovation_counter: 0,
            num_inputs: inputs,
            num_outputs: outputs,
            members: Vec::new(),
        };

        // initial topology is [inputs] + bias -> [outputs]
        // with a connection from the bias neuron to each output
        let bias_idx = inputs + outputs;
        let first_output_idx = inputs;

        let initial_innovation_numbers = (0..outputs)
            .map(|i| pop.gen_innovation_number_for(bias_idx, first_output_idx+i))
            .collect::<Vec<_>>();

        for _ in 0..pop_size {
            let mut genome = Genome::empty();

            for i in 0..outputs {
                genome.add_gene(Gene {
                    innovation_number: initial_innovation_numbers[i],
                    neuron_from: bias_idx,
                    neuron_to: first_output_idx + i,
                    weight: rand::thread_rng().gen_range(-1.0..1.0),
                    enabled: true,
                    from_is_bias: true,
                });
            }

            pop.members.push(genome);
        }

        pop
    }

    fn gen_innovation_number_for(&mut self, from: usize, to: usize) -> usize {
        if let Some(innov_number) = self.innovation_storage.get(&(from, to)) {
            *innov_number
        } else {
            let innov_number = self.innovation_counter;
            self.innovation_counter += 1;
            self.innovation_storage.insert((from, to), innov_number);
            innov_number
        }
    }

    fn get_phenotype(&self, genome: &Genome) -> Ctrnn {
        let num_neurons = genome.max_neuron_id + 1;
        let mut ctrnn = Ctrnn::new(num_neurons, self.num_inputs, self.num_outputs);

        for (_, gene) in genome.genes.iter() {
            if gene.enabled {
                ctrnn.wji.mut_data()[gene.neuron_to * num_neurons + gene.neuron_from] = gene.weight;
                if gene.from_is_bias {
                    ctrnn.theta.mut_data()[gene.neuron_from] = 1.0f64;
                }
            }
        }

        ctrnn
    }

    fn crossover(&self, lhs: &Genome, rhs: &Genome) -> Genome {
        let mut child = Genome::empty();

        for (l, r) in full_sorted_outer_join(lhs.genes.values(), rhs.genes.values(), |a, b| {
            a.innovation_number.cmp(&b.innovation_number)
        }) {
            match (l, r) {
                // TODO: preset chance that a gene is disabled if it is disabled in either parent
                (Some(l), Some(r)) => {
                    if lhs.fitness > rhs.fitness {
                        child.add_gene(l.clone());
                    } else {
                        child.add_gene(r.clone());
                    }
                }
                (Some(l), None) => {
                    if lhs.fitness > rhs.fitness {
                        child.add_gene(l.clone());
                    }
                }
                (None, Some(r)) => {
                    if rhs.fitness > lhs.fitness {
                        child.add_gene(r.clone());
                    }
                }
                (None, None) => unreachable!(),
            }
        }

        child.max_neuron_id = child
            .genes
            .values()
            .fold(0, |acc, gene| acc.max(gene.neuron_from).max(gene.neuron_to))
            .max(self.num_inputs + self.num_outputs - 1);

        child
    }

    fn mutate(
        &mut self,
        genome: &mut Genome,
    ) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < MUTATE_GENOME_WEIGHT_CHANGE {
            for gene in genome.genes.values_mut() {
                if rng.gen::<f64>() < MUTATE_GENE_WEIGHT_CHANGE {
                    gene.weight += rng.gen_range(-0.1..0.1);
                }
            }
        }

        for gene in genome.genes.values_mut() {
            if rng.gen::<f64>() < MUTATE_GENE_TOGGLE_EXPRESSION {
                gene.enabled = !gene.enabled;
            }
        }

        if rng.gen::<f64>() < MUTATE_GENOME_ADD_CONNECTION {
            let neuron_from = rng.gen_range(0..=genome.max_neuron_id);
            let neuron_to = rng.gen_range(0..=genome.max_neuron_id);

            let innovation_number = self.gen_innovation_number_for(neuron_from, neuron_to);
            let weight = rng.gen_range(-1.0..1.0);

            genome.add_gene(Gene {
                innovation_number,
                neuron_from,
                neuron_to,
                weight,
                enabled: true,
                from_is_bias: false,
            });
        }

        if rng.gen::<f64>() < MUTATE_GENOME_ADD_NEURON && genome.genes.len() > 0 {
            let gene_idx = rng.gen_range(0..genome.genes.len());
            let gene_list = genome.genes.values().collect::<Vec<_>>();

            let gene_to_split = gene_list[gene_idx].clone();

            genome
                .genes
                .get_mut(&gene_to_split.innovation_number)
                .unwrap()
                .enabled = false;

            let new_neuron_id = genome.max_neuron_id + 1;

            genome.add_gene(Gene {
                innovation_number: self.gen_innovation_number_for(gene_to_split.neuron_from, new_neuron_id),
                neuron_from: gene_to_split.neuron_from,
                neuron_to: new_neuron_id,
                weight: 1.0,
                enabled: true,
                from_is_bias: false,
            });

            genome.add_gene(Gene {
                innovation_number: self.gen_innovation_number_for(new_neuron_id, gene_to_split.neuron_to),
                neuron_from: new_neuron_id,
                neuron_to: gene_to_split.neuron_to,
                weight: gene_to_split.weight,
                enabled: true,
                from_is_bias: false,
            });
        }
    }

    fn evolve(&mut self) {
        // make sure that the same connection is always represented by the same innovation number
        // if it originates from the same evolution step
        let mut rng = rand::thread_rng();
        self.members
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut new_population = Vec::new();
        let num_elites = 1 + self.members.len() / 10;
        for i in 0..num_elites {
            new_population.push(self.members[i].clone());
        }

        while new_population.len() < self.members.len() {
            let parent1 = rng.gen_range(0..self.members.len());
            let parent2 = rng.gen_range(0..self.members.len());
            let mut child = self.crossover(&self.members[parent1], &self.members[parent2]);
            self.mutate(&mut child);
            new_population.push(child);
        }

        self.members = new_population;
    }
}

#[derive(Clone, Debug)]
struct Gene {
    innovation_number: usize,
    neuron_from: usize,
    neuron_to: usize,
    weight: f64,
    enabled: bool,
    from_is_bias: bool,
}

#[derive(Debug, Clone)]
struct Genome {
    genes: BTreeMap<usize, Gene>,
    max_neuron_id: usize,
    fitness: f64,
}

impl Genome {
    fn empty() -> Genome {
        Genome {
            genes: BTreeMap::new(),
            max_neuron_id: 0,
            fitness: 0.0,
        }
    }

    fn add_gene(&mut self, gene: Gene) {
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

        self.genes.insert(gene.innovation_number, gene);
    }

    fn print_dot(&self) {
        for i in 0..=self.max_neuron_id {
            println!("n{}", i);
        }

        for gene in self.genes.values() {
            if gene.enabled {
                println!(
                    "n{} -> n{} [label=\"{}\"]",
                    gene.neuron_from, gene.neuron_to, gene.weight
                );
            }
        }
    }
}

fn main() {
    let mut population = Population::new(2, 1, 50);
    let xor_results = vec![
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 0.0),
    ];

    for _ in 0..2000 {
        for genome_idx in 0..population.members.len() {
            let mut network = population.get_phenotype(&population.members[genome_idx]);
            let genome = &mut population.members[genome_idx];
            genome.fitness = 0.0;
            for ((lhs, rhs), expected) in xor_results.iter() {
                network.update(0.1, &vec![*lhs, *rhs]);
                let output = network.get_outputs()[0];
                genome.fitness += 1.0 - (output - expected).abs();
            }
        }
        population.evolve();
    }

    population
        .members
        .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    let genome = &population.members[0];
    let mut i = population.get_phenotype(&genome);
    dbg!(genome);

    for ((inputs, outputs), _) in xor_results {
        i.update(0.1, &vec![inputs, outputs]);
        println!("{} XOR {} = {}", inputs, outputs, i.get_outputs()[0]);
    }

    genome.print_dot();
}
