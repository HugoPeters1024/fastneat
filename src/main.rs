mod ctrnn;
mod genome;
mod params;
mod population;
mod specie;

use population::*;

use crate::params::Settings;

fn main() {
    let settings = Settings {
        num_inputs: 2,
        num_outputs: 1,
        population_size: 200,
        target_species: 5,
    };
    let mut population = Population::new(&settings);

    let xor_results = vec![
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 0.0),
    ];

    for _ in 0..600 {
        for genome_idx in 0..population.members.len() {
            let mut network = population.get_phenotype(&population.members[genome_idx]);
            let genome = &mut population.members[genome_idx];
            genome.fitness = 0.0;
            for ((lhs, rhs), expected) in xor_results.iter() {
                for _ in 0..10 {
                    network.update(0.2, &vec![*lhs, *rhs]);
                }
                let output = network.get_outputs()[0];
                genome.fitness += 1.0 - ((output - expected).abs()).powi(2);
            }
        }
        population.evolve();
    }

    let genome = &population.get_winner();
    let mut i = population.get_phenotype(&genome);
    dbg!(genome);

    for ((inputs, outputs), _) in xor_results {
        for _ in 0..10 {
            i.update(0.2, &vec![inputs, outputs]);
        }
        println!("{} XOR {} = {}", inputs, outputs, i.get_outputs()[0]);
    }

    genome.print_dot();
}
