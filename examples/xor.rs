use fastneat::params::*;
use fastneat::population::*;

const XOR_RESULTS: [((f64, f64), f64); 4] = [
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
];

fn main() {
    let settings = Settings {
        num_inputs: 2,
        num_outputs: 1,
        population_size: 80,
        target_species: 3,
        parameters: Parameters {
            mutate_genome_add_connection: 0.5,
            mutate_genome_add_neuron: 0.01,
            mutate_genome_add_bias_neuron: 0.0,
            specie_greediness_exponent: 3.5,
            specie_dropoff_age: 15,
            enable_elitism: true,
            mutate_genome_tau_change: 0.0,
            ..Default::default()
        },
    };
    let mut population = Population::new(&settings);

    for _ in 0..1700 {
        eval_population(&mut population);
        if population.get_winner().fitness > 3.99 {
            break;
        }
        population.evolve();
    }

    eval_population(&mut population);
    let genome = &population.get_winner();
    dbg!(genome);

    for ((inputs, outputs), _) in XOR_RESULTS {
        let mut i = population.get_phenotype(&genome);
        for _ in 0..10 {
            i.update(0.2, &vec![inputs, outputs]);
        }
        println!("{} XOR {} = {}", inputs, outputs, i.get_outputs()[0]);
    }

    genome.print_dot();
}

fn eval_population(population: &mut Population) {
    for genome_idx in 0..population.members.len() {
        (&mut population.members[genome_idx]).fitness = 0.0;
        for ((lhs, rhs), expected) in XOR_RESULTS.iter() {
            let mut network = population.get_phenotype(&population.members[genome_idx]);
            for _ in 0..10 {
                network.update(0.2, &vec![*lhs, *rhs]);
            }
            let output = network.get_outputs()[0];
            (&mut population.members[genome_idx]).fitness += 1.0 - (output - expected).abs();
        }
    }
}
