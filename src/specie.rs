use crate::params::SPECIE_DROPOFF_AGE;

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

    pub fn should_die(&self) -> bool {
        self.age_last_improvement >= SPECIE_DROPOFF_AGE
    }
}
