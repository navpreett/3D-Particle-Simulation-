use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use cgmath::prelude::*;
use encase::ShaderType;
use rayon::prelude::*;

#[derive(Clone, Copy, ShaderType)]
pub struct Particle {
    pub position: cgmath::Vector3<f32>,
    pub velocity: cgmath::Vector3<f32>,
    pub id: u32,
}

#[derive(Default)]
pub struct Particles {
    pub world_size: f32,
    pub current_particles: Vec<Particle>,
    pub previous_particles: Vec<Particle>,
    pub id_count: u32,
    pub attraction_matrix: Vec<f32>,
    pub colors: Vec<cgmath::Vector3<f32>>,
    pub friction_half_time: f32,
    pub force_scale: f32,
    pub particle_effect_radius: f32,
}

impl Particles {
    fn cell_coord(&self, position: cgmath::Vector3<f32>) -> cgmath::Vector3<isize> {
        cgmath::vec3(
            (position.x / self.particle_effect_radius) as isize,
            (position.y / self.particle_effect_radius) as isize,
            (position.z / self.particle_effect_radius) as isize,
        )
    }

    fn hash(cell: cgmath::Vector3<isize>) -> usize {
        let mut hasher = DefaultHasher::new();
        cell.x.hash(&mut hasher);
        cell.y.hash(&mut hasher);
        cell.z.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn calculate_forces(&self, particle: &Particle, other_particles: &[Particle]) -> cgmath::Vector3<f32> {
        let mut total_force = cgmath::Vector3::zero();
        for other_particle in other_particles {
            let relative_position = other_particle.position - particle.position;
            let sqr_distance = relative_position.magnitude2();
            if sqr_distance > 0.0
                && sqr_distance < self.particle_effect_radius * self.particle_effect_radius
            {
                let distance = sqr_distance.sqrt();
                let f = Self::force(distance, self.attraction_matrix[(particle.id * self.id_count + other_particle.id) as usize]);
                total_force += relative_position / distance * f;
            }
        }
        total_force
    }

    fn force(r: f32, attraction: f32) -> f32 {
        const BETA: f32 = 0.3;
        if r < BETA {
            r / BETA - 1.0
        } else if BETA < r && r < 1.0 {
            attraction * (1.0 - (2.0 * r - 1.0 - BETA).abs() / (1.0 - BETA))
        } else {
            0.0
        }
    }

    fn update_particle(&self, particle: &Particle, ts: f32) -> Particle {
        let mut particle = *particle;
        particle.velocity += Self::calculate_forces(self, &particle, &self.previous_particles) * self.force_scale * self.particle_effect_radius * ts;
        particle
    }

    fn apply_friction(&self, particle: &mut Particle, ts: f32) {
        let friction_constant = 0.5f32.powf(self.friction_half_time);
        particle.velocity -= particle.velocity * friction_constant * ts;
        particle.position += particle.velocity * ts;
        if particle.position.x > self.world_size * 0.5 {
            particle.position.x -= self.world_size;
        }
        if particle.position.x < -self.world_size * 0.5 {
            particle.position.x += self.world_size;
        }
        if particle.position.y > self.world_size * 0.5 {
            particle.position.y -= self.world_size;
        }
        if particle.position.y < -self.world_size * 0.5 {
            particle.position.y += self.world_size;
        }
        if particle.position.z > self.world_size * 0.5 {
            particle.position.z -= self.world_size;
        }
        if particle.position.z < -self.world_size * 0.5 {
            particle.position.z += self.world_size;
        }
    }

    pub fn update(&mut self, ts: f32) {
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);
    
        let hash_table_length = self.current_particles.len();
        let hash_table: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(hash_table_length + 1)
            .collect();
    
        self.current_particles.par_iter().for_each(|sphere| {
            let index = Self::hash(Self::cell_coord(self, sphere.position)) % hash_table_length;
            hash_table[index].fetch_add(1, Relaxed);
        });
    
        for i in 1..hash_table.len() {
            hash_table[i].fetch_add(hash_table[i - 1].load(Relaxed), Relaxed);
        }
    
        let particle_indices: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(self.current_particles.len())
            .collect();
        self.current_particles
            .par_iter()
            .enumerate()
            .for_each(|(i, sphere)| {
                let index = Self::hash(Self::cell_coord(self, sphere.position)) % hash_table_length;
                let index = hash_table[index].fetch_sub(1, Relaxed);
                particle_indices[index - 1].store(i, Relaxed);
            });
    
        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
    
        let updated_particles: Vec<_> = self.previous_particles
            .par_iter()
            .map(|particle| {
                let updated_particle = Self::update_particle(self, particle, ts);
                let mut updated_particle = updated_particle;
                Self::apply_friction(self, &mut updated_particle, ts);
                updated_particle
            })
            .collect();
    
        self.current_particles = updated_particles;
    }
}    