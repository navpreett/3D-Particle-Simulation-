use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use cgmath::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Particle {
    pub position: cgmath::Vector3<f32>,
    pub velocity: cgmath::Vector3<f32>,
    pub id: usize,
}

#[derive(Default)]
pub struct Particles {
    pub world_size: f32,
    pub current_particles: Vec<Particle>,
    pub previous_particles: Vec<Particle>,
    pub id_count: usize,
    pub attraction_matrix: Vec<f32>,
    pub colors: Vec<cgmath::Vector3<f32>>,
    pub friction_half_time: f32,
    pub force_scale: f32,
    pub particle_effect_radius: f32,
}

impl Particles {
    pub fn update(&mut self, ts: f32) {
        self.apply_forces(ts);
        
        self.update_positions(ts);
    }

    fn apply_forces(&mut self, ts: f32) {
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);

        let (hash_table, particle_indices) = self.build_spatial_hash_table();
        let hash_table_length = self.current_particles.len();

        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();

        let effect_radius_squared = self.particle_effect_radius * self.particle_effect_radius;
        let force_scale_factor = self.force_scale * self.particle_effect_radius * ts;
        let world_size = self.world_size;
        let particle_effect_radius = self.particle_effect_radius;
        let attraction_matrix = &self.attraction_matrix;
        let id_count = self.id_count;
        let previous_particles = &self.previous_particles;

        let updated_particles: Vec<_> = previous_particles
            .par_iter()
            .map(|&particle| {
                compute_particle_update(
                    particle,
                    &hash_table,
                    &particle_indices,
                    hash_table_length,
                    effect_radius_squared,
                    force_scale_factor,
                    world_size,
                    particle_effect_radius,
                    attraction_matrix,
                    id_count,
                    previous_particles,
                )
            })
            .collect();

        self.current_particles.extend(updated_particles);
    }

    fn update_positions(&mut self, ts: f32) {
        let friction_factor = 1.0 - 0.5f32.powf(ts / self.friction_half_time);
        let half_world_size = self.world_size * 0.5;
        let world_size = self.world_size;
        
        self.current_particles.par_iter_mut().for_each(|particle| {
            particle.velocity *= friction_factor;
            
            particle.position += particle.velocity * ts;
            
            particle.position = apply_periodic_boundary(
                particle.position, 
                half_world_size, 
                world_size
            );
        });
    }

    fn build_spatial_hash_table(&self) -> (Vec<AtomicUsize>, Vec<AtomicUsize>) {
        let hash_table_length = self.current_particles.len();
        let particle_effect_radius = self.particle_effect_radius;
        
        let hash_table: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(hash_table_length + 1)
            .collect();

        self.current_particles.par_iter().for_each(|particle| {
            let index = hash_vector(cell_coord(particle.position, particle_effect_radius)) % hash_table_length;
            hash_table[index].fetch_add(1, Relaxed);
        });

        let mut prev_count = 0;
        for i in 0..hash_table.len() {
            let current = hash_table[i].load(Relaxed);
            hash_table[i].store(prev_count, Relaxed);
            prev_count += current;
        }

        let particle_indices: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(self.current_particles.len())
            .collect();

        self.current_particles
            .par_iter()
            .enumerate()
            .for_each(|(i, particle)| {
                let index = hash_vector(cell_coord(particle.position, particle_effect_radius)) % hash_table_length;
                let position = hash_table[index].fetch_add(1, Relaxed);
                particle_indices[position].store(i, Relaxed);
            });

        (hash_table, particle_indices)
    }
}

#[inline]
fn cell_coord(v: cgmath::Vector3<f32>, particle_effect_radius: f32) -> cgmath::Vector3<isize> {
    cgmath::vec3(
        (v.x / particle_effect_radius) as isize,
        (v.y / particle_effect_radius) as isize,
        (v.z / particle_effect_radius) as isize,
    )
}

#[inline]
fn compute_force(r: f32, attraction: f32) -> f32 {
    const BETA: f32 = 0.3;
    if r < BETA {
        r / BETA - 1.0
    } else if r < 1.0 {
        attraction * (1.0 - (2.0 * r - 1.0 - BETA).abs() / (1.0 - BETA))
    } else {
        0.0
    }
}

#[inline]
fn hash_vector(v: cgmath::Vector3<isize>) -> usize {
    let mut hasher = DefaultHasher::new();
    v.x.hash(&mut hasher);
    v.y.hash(&mut hasher);
    v.z.hash(&mut hasher);
    hasher.finish() as usize
}

#[inline]
fn apply_periodic_boundary(
    mut position: cgmath::Vector3<f32>, 
    half_world_size: f32, 
    world_size: f32
) -> cgmath::Vector3<f32> {
    if position.x > half_world_size {
        position.x -= world_size;
    } else if position.x < -half_world_size {
        position.x += world_size;
    }
    
    if position.y > half_world_size {
        position.y -= world_size;
    } else if position.y < -half_world_size {
        position.y += world_size;
    }
    
    if position.z > half_world_size {
        position.z -= world_size;
    } else if position.z < -half_world_size {
        position.z += world_size;
    }
    
    position
}

#[inline]
fn compute_particle_update(
    mut particle: Particle,
    hash_table: &[AtomicUsize],
    particle_indices: &[AtomicUsize],
    hash_table_length: usize,
    effect_radius_squared: f32,
    force_scale_factor: f32,
    world_size: f32,
    particle_effect_radius: f32,
    attraction_matrix: &[f32],
    id_count: usize,
    previous_particles: &[Particle],
) -> Particle {
    let mut total_force = cgmath::Vector3::zero();
    
    for x_offset in -1..=1 {
        for y_offset in -1..=1 {
            for z_offset in -1..=1 {
                let offset = cgmath::vec3(x_offset as f32, y_offset as f32, z_offset as f32) 
                    * world_size;
                
                process_neighboring_cells(
                    &mut total_force,
                    particle,
                    offset,
                    hash_table,
                    particle_indices,
                    hash_table_length,
                    effect_radius_squared,
                    particle_effect_radius,
                    attraction_matrix,
                    id_count,
                    previous_particles,
                );
            }
        }
    }
    
    particle.velocity += total_force * force_scale_factor;
    particle
}

#[inline]
fn process_neighboring_cells(
    total_force: &mut cgmath::Vector3<f32>,
    particle: Particle,
    offset: cgmath::Vector3<f32>,
    hash_table: &[AtomicUsize],
    particle_indices: &[AtomicUsize],
    hash_table_length: usize,
    effect_radius_squared: f32,
    particle_effect_radius: f32,
    attraction_matrix: &[f32],
    id_count: usize,
    previous_particles: &[Particle],
) {
    let cell = cell_coord(particle.position + offset, particle_effect_radius);

    for x_cell_offset in -1isize..=1 {
        for y_cell_offset in -1isize..=1 {
            for z_cell_offset in -1isize..=1 {
                let neighbor_cell = cell + cgmath::vec3(
                    x_cell_offset,
                    y_cell_offset,
                    z_cell_offset,
                );

                let cell_hash = hash_vector(neighbor_cell) % hash_table_length;
                let start = hash_table[cell_hash].load(Relaxed);
                let end = hash_table[cell_hash + 1].load(Relaxed);
                
                for particle_index in &particle_indices[start..end] {
                    let other_particle = &previous_particles[particle_index.load(Relaxed)];
                    
                    let relative_position = other_particle.position - particle.position + offset;
                    let sqr_distance = relative_position.magnitude2();
                    
                    if sqr_distance > 0.0 && sqr_distance < effect_radius_squared {
                        let distance = sqr_distance.sqrt();
                        let attraction = attraction_matrix[particle.id * id_count + other_particle.id];
                        let force = compute_force(distance / particle_effect_radius, attraction);
                        *total_force += relative_position / distance * force;
                    }
                }
            }
        }
    }
}