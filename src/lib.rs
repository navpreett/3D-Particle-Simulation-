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

fn hash_position(position: cgmath::Vector3<isize>) -> usize {
    let cgmath::Vector3 { x, y, z } = position;
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    y.hash(&mut hasher);
    z.hash(&mut hasher);
    hasher.finish() as usize
}

fn calculate_force(r: f32, attraction: f32) -> f32 {
    const BETA: f32 = 0.3;
    if r < BETA {
        r / BETA - 1.0
    } else if BETA < r && r < 1.0 {
        attraction * (1.0 - (2.0 * r - 1.0 - BETA).abs() / (1.0 - BETA))
    } else {
        0.0
    }
}

fn position_to_cell(position: cgmath::Vector3<f32>, particle_effect_radius: f32) -> cgmath::Vector3<isize> {
    cgmath::vec3(
        (position.x / particle_effect_radius) as isize,
        (position.y / particle_effect_radius) as isize,
        (position.z / particle_effect_radius) as isize,
    )
}

fn apply_boundary_conditions(position: cgmath::Vector3<f32>, world_size: f32) -> cgmath::Vector3<f32> {
    let mut result = position;
    let half_size = world_size * 0.5;
    
    if result.x > half_size { result.x -= world_size; }
    if result.x < -half_size { result.x += world_size; }
    
    if result.y > half_size { result.y -= world_size; }
    if result.y < -half_size { result.y += world_size; }
    
    if result.z > half_size { result.z -= world_size; }
    if result.z < -half_size { result.z += world_size; }
    
    result
}

impl Particles {
    pub fn update(&mut self, ts: f32) {
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);
        
        let particle_effect_radius = self.particle_effect_radius;
        let world_size = self.world_size;
        let id_count = self.id_count;
        let attraction_matrix = &self.attraction_matrix;
        
        let hash_table_length = self.current_particles.len();
        let hash_table: Vec<_> = (0..=hash_table_length)
            .into_par_iter()
            .map(|_| AtomicUsize::new(0))
            .collect();
        
        self.current_particles.par_iter().for_each(|particle| {
            let cell = position_to_cell(particle.position, particle_effect_radius);
            let index = hash_position(cell) % hash_table_length;
            hash_table[index].fetch_add(1, Relaxed);
        });
        
        for i in 1..hash_table.len() {
            hash_table[i].fetch_add(hash_table[i - 1].load(Relaxed), Relaxed);
        }
        
        let particle_indices: Vec<_> = (0..self.current_particles.len())
            .into_par_iter()
            .map(|_| AtomicUsize::new(0))
            .collect();
            
        self.current_particles.par_iter().enumerate().for_each(|(i, particle)| {
            let cell = position_to_cell(particle.position, particle_effect_radius);
            let index = hash_position(cell) % hash_table_length;
            let insert_index = hash_table[index].fetch_sub(1, Relaxed) - 1;
            particle_indices[insert_index].store(i, Relaxed);
        });
        
        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();
        
        let force_scale = self.force_scale * particle_effect_radius * ts;
        let previous_particles = &self.previous_particles;
        let current_particles_result: Vec<Particle> = previous_particles.par_iter().map(|&particle| {
            let mut updated_particle = particle;
            let mut total_force = cgmath::Vector3::zero();
            
            for x_offset in -1..=1 {
                for y_offset in -1..=1 {
                    for z_offset in -1..=1 {
                        let offset = cgmath::vec3(x_offset as f32, y_offset as f32, z_offset as f32) * world_size;
                        let base_cell = position_to_cell(particle.position + offset, particle_effect_radius);
                        
                        for x_cell_offset in -1isize..=1 {
                            for y_cell_offset in -1isize..=1 {
                                for z_cell_offset in -1isize..=1 {
                                    let cell = base_cell + cgmath::vec3(x_cell_offset, y_cell_offset, z_cell_offset);
                                    let index = hash_position(cell) % hash_table_length;
                                    
                                    for i in hash_table[index].load(Relaxed)..hash_table[index + 1].load(Relaxed) {
                                        let other_index = particle_indices[i].load(Relaxed);
                                        let other_particle = &previous_particles[other_index];
                                        
                                        let relative_position = other_particle.position - particle.position + offset;
                                        let sqr_distance = relative_position.magnitude2();
                                        
                                        if sqr_distance > 0.0 && sqr_distance < particle_effect_radius * particle_effect_radius {
                                            let distance = sqr_distance.sqrt();
                                            let attraction_idx = (particle.id * id_count + other_particle.id) as usize;
                                            let f = calculate_force(distance, attraction_matrix[attraction_idx]);
                                            total_force += relative_position / distance * f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            updated_particle.velocity += total_force * force_scale;
            updated_particle
        }).collect();
        
        self.current_particles.extend(current_particles_result);
        
        let friction_factor = 0.5f32.powf(self.friction_half_time) * ts;
        self.current_particles.par_iter_mut().for_each(|particle| {
            particle.velocity -= particle.velocity * friction_factor;
            
            particle.position += particle.velocity * ts;
            
            particle.position = apply_boundary_conditions(particle.position, world_size);
        });
    }
}