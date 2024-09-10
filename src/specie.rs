use crate::params::Parameters;

#[derive(Debug)]
pub struct Specie {
    pub rep_idx: usize,
    pub max_fitness_seen: f64,
    pub age: usize,
    pub age_last_improvement: usize,
}

impl Specie {
    pub fn from_rep(rep_id: usize) -> Self {
        Specie {
            rep_idx: rep_id,
            max_fitness_seen: 0.0,
            age: 0,
            age_last_improvement: 0,
        }
    }

    pub fn should_die(&self, params: &Parameters) -> bool {
        self.age_last_improvement >= params.specie_dropoff_age
    }
}
