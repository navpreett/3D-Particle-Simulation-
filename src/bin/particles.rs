use macroquad::prelude::*;
use super::particle::Particle;

pub struct Particles {
    pub world_size: f32,
    pub id_count: usize,
    pub colors: Vec<Color>,
    pub attraction_matrix: Vec<f32>,
    pub particle_effect_radius: f32,
    pub friction_half_time: f32,
    pub force_scale: f32,
    pub current_particles: Vec<Particle>,
    pub previous_particles: Vec<Particle>,
}

impl Particles {
    pub fn update(&mut self, delta_time: f32) {
        // tba
    }
}