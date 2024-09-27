use rand::Rng;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::ctrnn::Ctrnn;
use crate::genome::*;
use crate::params::*;
use crate::specie::Specie;

pub struct Population {
    _next_innovation_id: usize,
    _innovation_cache: HashMap<(usize, usize), usize>,
    _next_specie_id: usize,
    settings: Settings,
    pub members: Vec<Genome>,
    // specie id -> Specie
    species: HashMap<usize, Specie>,
    compatility_threshold: f64,
    generation: usize,
}

impl Population {
    pub fn new(settings: &Settings) -> Population {
        let mut pop = Population {
            _next_innovation_id: 0,
            _innovation_cache: HashMap::new(),
            _next_specie_id: 0,
            settings: settings.clone(),
            members: Vec::new(),
            species: HashMap::new(),
            compatility_threshold: settings.parameters.specie_threshold_initial,
            generation: 0,
        };

        for _ in 0..settings.population_size {
            let new_member = pop.spawn_new_genome();
            pop.members.push(new_member);
        }

        pop.speciate();
        pop
    }

    fn next_innovation_number(&mut self, neuron_from: usize, neuron_to: usize) -> usize {
        if let Some(cached) = self._innovation_cache.get(&(neuron_from, neuron_to)) {
            return *cached;
        }

        let ret = self._next_innovation_id;
        self._next_innovation_id += 1;
        self._innovation_cache.insert((neuron_from, neuron_to), ret);
        return ret;
    }

    fn next_specie_id(&mut self) -> usize {
        let ret = self._next_specie_id;
        self._next_specie_id += 1;
        return ret;
    }

    fn spawn_new_genome(&mut self) -> Genome {
        Genome::empty(&self.settings)
    }

    fn speciate(&mut self) {
        // collect in which specie existing organisms are.
        let mut members_by_species: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, member) in self.members.iter().enumerate() {
            if let Some(specie) = member.specie_idx {
                members_by_species.entry(specie).or_default().push(i);
            }
        }

        // remove any species without members
        self.species
            .retain(|specie_id, _| members_by_species.contains_key(specie_id));

        // unassign members from their species
        for member in self.members.iter_mut() {
            member.specie_idx = None;
        }

        // select a new rep for each specie
        for (specie_id, specie) in self.species.iter_mut() {
            let specie_members = &members_by_species[specie_id];
            debug_assert!(!specie_members.is_empty());

            let specie_winner: usize = specie_members
                .iter()
                .max_by(|l, r| {
                    self.members[**l]
                        .fitness
                        .partial_cmp(&self.members[**r].fitness)
                        .unwrap()
                })
                .unwrap()
                .clone();

            specie.rep_idx = specie_winner;
            // also let the rep itself know it is again part of the species
            self.members[specie_winner].specie_idx = Some(*specie_id);
        }

        // for each member, compare to each specie to see if it belongs there.
        // if it doesn't belong in any, create a new specie with that individual
        // as the rep.
        'member_loop: for member_idx in 0..self.members.len() {
            if self.members[member_idx].specie_idx.is_some() {
                continue;
            }

            for (specie_idx, specie) in self.species.iter() {
                let rep = &self.members[specie.rep_idx];
                if self.members[member_idx].compatibility(
                    &rep,
                    self.settings.parameters.compat_mismatch_genes_factor,
                    self.settings.parameters.compat_mismatch_weight_factor,
                ) < self.compatility_threshold
                {
                    self.members[member_idx].specie_idx = Some(*specie_idx);
                    continue 'member_loop;
                }
            }

            // no matching specie was found, this member is a new specie
            let new_specie_id = self.next_specie_id();
            self.species
                .insert(new_specie_id, Specie::from_rep(member_idx));
            self.members[member_idx].specie_idx = Some(new_specie_id);
        }

        self.assert_species_consisent();
    }

    pub fn get_phenotype(&self, genome: &Genome) -> Ctrnn {
        let num_neurons = genome.get_num_neurons();
        let mut ctrnn = Ctrnn::new(
            num_neurons,
            self.settings.num_inputs,
            self.settings.num_outputs,
            self.settings.parameters.activation_function.clone(),
        );

        for (neuron_id, neuron) in genome.neurons.iter() {
            ctrnn.tau.mut_data()[*neuron_id] = neuron.tau;
            if neuron.is_bias {
                ctrnn.theta.mut_data()[*neuron_id] = 1.0;
            }
        }

        for (_, gene) in genome.genes.iter() {
            if gene.enabled {
                ctrnn.wji.mut_data()[gene.neuron_to * num_neurons + gene.neuron_from] = gene.weight;
            }
        }

        ctrnn
    }

    fn mutate(&mut self, genome: &mut Genome) {
        let mut rng = rand::thread_rng();
        let params = self.settings.parameters.clone();

        if rng.gen::<f64>() < params.mutate_genome_weight_change {
            for gene in genome.genes.values_mut() {
                if rng.gen::<f64>() < params.mutate_gene_weight_change {
                    gene.weight += rng.gen_range(
                        -params.mutate_gene_nudge_factor..params.mutate_gene_nudge_factor,
                    );
                }
            }
        }

        if rng.gen::<f64>() < params.mutate_genome_tau_change {
            for neuron in genome.neurons.values_mut() {
                if rng.gen::<f64>() < params.mutate_neuron_tau_change {
                    let nudge = rng.gen_range(
                        -params.mutate_neuron_tau_nudge_factor
                            ..params.mutate_neuron_tau_nudge_factor,
                    );
                    neuron.tau = (neuron.tau + nudge).max(0.005);
                }
            }
        }

        for gene in genome.genes.values_mut() {
            if rng.gen::<f64>() < params.mutate_gene_toggle_expression {
                gene.enabled = !gene.enabled;
            }
        }

        if rng.gen::<f64>() < params.mutate_genome_add_connection {
            let neuron_from = genome.sample_neuron_id();
            let neuron_to = genome.sample_neuron_id();
            let innovation_number = self.next_innovation_number(neuron_from, neuron_to);
            let weight = rng.gen_range(-10.0..10.0);

            genome.add_gene(Gene {
                innovation_number,
                neuron_from,
                neuron_to,
                weight,
                enabled: true,
            });
        }

        if rng.gen::<f64>() < params.mutate_genome_add_neuron && genome.genes.len() > 0 {
            let new_neuron_id = genome.get_num_neurons();
            let gene_to_split = genome.sample_gene_mut();

            // disable the direct connection
            gene_to_split.enabled = false;

            // A new connection to the new neuron with weight 1
            let connection_to = Gene {
                innovation_number: self
                    .next_innovation_number(gene_to_split.neuron_from, new_neuron_id),
                neuron_from: gene_to_split.neuron_from,
                neuron_to: new_neuron_id,
                weight: 1.0,
                enabled: true,
            };

            // A new connection from the new neuron with the orignal weight
            let connection_from = Gene {
                innovation_number: self
                    .next_innovation_number(new_neuron_id, gene_to_split.neuron_to),
                neuron_from: new_neuron_id,
                neuron_to: gene_to_split.neuron_to,
                weight: gene_to_split.weight,
                enabled: true,
            };

            genome.add_gene(connection_to);
            genome.add_gene(connection_from);
        }

        if rng.gen::<f64>() < params.mutate_genome_add_bias_neuron {
            let new_neuron_id = genome.get_num_neurons();
            let connect_to = genome.sample_neuron_id();

            let connection = Gene {
                innovation_number: self.next_innovation_number(new_neuron_id, connect_to),
                neuron_from: new_neuron_id,
                neuron_to: connect_to,
                weight: 1.0,
                enabled: true,
            };

            genome.add_neuron(
                new_neuron_id,
                Neuron {
                    tau: 0.1,
                    is_bias: true,
                },
            );
            genome.add_gene(connection);
        }
    }

    pub fn crossover(&self, lhs: &Genome, rhs: &Genome) -> Genome {
        let mut rng = rand::thread_rng();
        let mut child = Genome::empty(&self.settings);

        for (l, r) in full_sorted_outer_join(lhs.genes.values(), rhs.genes.values(), |a, b| {
            a.innovation_number.cmp(&b.innovation_number)
        }) {
            match (l, r) {
                (Some(l), Some(r)) => {
                    let mut new_gene = if rng.gen::<bool>() {
                        l.clone()
                    } else {
                        r.clone()
                    };

                    if (!l.enabled || !r.enabled) && rng.gen::<f32>() < 0.5 {
                        new_gene.enabled = false
                    }

                    child.add_gene(new_gene);
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

        // ensure we keep the tau values of the fittest parent
        for (neuron_idx, neuron) in &(if lhs.fitness > rhs.fitness { lhs } else { rhs }).neurons {
            if let Some(child_neuron) = child.neurons.get_mut(neuron_idx) {
                *child_neuron = neuron.clone();
            }
        }

        child
    }

    fn assert_species_consisent(&self) {
        for member in &self.members {
            if let Some(specie_idx) = member.specie_idx {
                debug_assert!(self.species.contains_key(&specie_idx));
            }
        }
    }

    pub fn evolve(&mut self) {
        self.assert_species_consisent();

        // update each specie's age and stats
        for specie in self.species.values_mut() {
            specie.age += 1;
            let rep = &self.members[specie.rep_idx];
            if rep.fitness > specie.max_fitness_seen {
                specie.max_fitness_seen = rep.fitness;
                specie.age_last_improvement = 0;
            } else {
                specie.age_last_improvement += 1;
            }
        }

        let mut new_population = Vec::new();

        let mut elite_idx = 0;
        let mut elite_fitness = f64::NEG_INFINITY;
        for (member_idx, member) in self.members.iter().enumerate() {
            if member.fitness > elite_fitness {
                elite_fitness = member.fitness;
                elite_idx = member_idx;
            }
        }

        if self.settings.parameters.enable_elitism {
            let mut elite = self.members[elite_idx].clone();
            elite.specie_idx = None;
            elite.fitness = 0.0;
            new_population.push(elite);
        }

        // collect in which specie existing organisms are.
        let mut members_by_species: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, member) in self.members.iter().enumerate() {
            members_by_species
                .entry(
                    member
                        .specie_idx
                        .expect("speciation should be done before evolving"),
                )
                .or_default()
                .push(i);
        }

        let adjusted_fitness: Vec<f64> = self
            .members
            .iter()
            .map(|m| {
                m.fitness
                    / members_by_species[&m.specie_idx.expect("speciation should be done")].len()
                        as f64
            })
            .collect();

        let adjusted_avg_fitness_by_species: HashMap<usize, f64> = members_by_species
            .iter()
            .map(|(specie_idx, specie_members)| {
                (
                    *specie_idx,
                    specie_members
                        .iter()
                        .map(|m| adjusted_fitness[*m])
                        .sum::<f64>()
                        / specie_members.len() as f64,
                )
            })
            .collect();

        let global_avg: f64 = adjusted_fitness.iter().sum::<f64>() / adjusted_fitness.len() as f64;

        let offspring_by_species: HashMap<usize, usize> = adjusted_avg_fitness_by_species
            .iter()
            .map(|(specie_idx, adjusted_fitness)| {
                (
                    *specie_idx,
                    if self.species[specie_idx].should_die(&self.settings.parameters) {
                        0
                    } else {
                        ((adjusted_fitness / global_avg)
                            * members_by_species[specie_idx].len() as f64)
                            .floor() as usize
                    },
                )
            })
            .collect();

        for (specie_idx, specie_members) in members_by_species.iter() {
            let mut all_fitnesses: Vec<(usize, f64)> = specie_members
                .iter()
                .map(|member_idx| (*member_idx, self.members[*member_idx].fitness))
                .collect();
            all_fitnesses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let total_fitness: f64 = all_fitnesses.iter().map(|x| x.1).sum();

            for _ in 0..offspring_by_species[specie_idx] {
                let parent1 = sample_parent(
                    &all_fitnesses,
                    total_fitness,
                    self.settings.parameters.specie_greediness_exponent,
                );
                let parent2 = sample_parent(
                    &all_fitnesses,
                    total_fitness,
                    self.settings.parameters.specie_greediness_exponent,
                );

                let mut child = self.crossover(&self.members[parent1], &self.members[parent2]);
                self.mutate(&mut child);
                child.specie_idx = Some(*specie_idx);
                new_population.push(child);
            }
        }

        while new_population.len() < self.members.len() {
            new_population.push(self.spawn_new_genome());
        }

        self.members = new_population.as_slice()[0..self.members.len()].to_vec();
        self.speciate();

        if self.species.len() < self.settings.target_species {
            self.compatility_threshold -= self.settings.parameters.specie_threshold_nudge_factor;
        }

        if self.species.len() > self.settings.target_species {
            self.compatility_threshold += self.settings.parameters.specie_threshold_nudge_factor;
        }

        self.generation += 1;

        println!(
            "gen = {}, nr_species = {}, pop_size = {}, threshold = {}, best_fitness = {}",
            self.generation,
            self.species.len(),
            self.members.len(),
            self.compatility_threshold,
            elite_fitness
        );
    }

    pub fn get_winner(&self) -> &Genome {
        return self
            .members
            .iter()
            .max_by(|lhs, rhs| lhs.fitness.partial_cmp(&rhs.fitness).unwrap())
            .unwrap();
    }
}

fn sample_parent(
    sorted_fitness: &Vec<(usize, f64)>,
    total_fitness: f64,
    greedy_exponent: f64,
) -> usize {
    let mut rng = rand::thread_rng();
    let sample = (rng.gen_range(0.0..1.0) as f64).powf(greedy_exponent) * total_fitness;
    let mut acc = 0.0;
    for (i, fitness) in sorted_fitness {
        acc += fitness;
        if acc >= sample {
            return *i;
        }
    }

    panic!(
        "was total_fitness not correct? sample = {}, total_fitness = {}",
        sample, total_fitness
    )
}

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
