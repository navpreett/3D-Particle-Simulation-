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
    pub friction: f32,
    pub force_scale: f32,
    pub particle_effect_radius: f32,
}

impl Particles {
    pub fn update(&mut self, ts: f32) {
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);

        #[inline]
        fn cell_coord(v: cgmath::Vector3<f32>, radius: f32) -> cgmath::Vector3<isize> {
            cgmath::vec3(
                (v.x / radius) as isize,
                (v.y / radius) as isize,
                (v.z / radius) as isize,
            )
        }

        #[inline]
        fn hash(v: cgmath::Vector3<isize>) -> usize {
            let mut hasher = DefaultHasher::new();
            v.x.hash(&mut hasher);
            v.y.hash(&mut hasher);
            v.z.hash(&mut hasher);
            hasher.finish() as usize
        }

        #[inline]
        fn calculate_force(r: f32, attraction: f32) -> f32 {
            const BETA: f32 = 0.3;
            if r < BETA {
                r / BETA - 1.0
            } else if r < 1.0 {
                attraction * (1.0 - (2.0 * r - 1.0 - BETA).abs() / (1.0 - BETA))
            } else {
                0.0
            }
        }

        let hash_table_length = self.current_particles.len();
        
        let hash_table: Vec<_> = (0..hash_table_length + 1)
            .map(|_| AtomicUsize::new(0))
            .collect();
            
        let cell_coords: Vec<_> = self.current_particles
            .par_iter()
            .map(|p| cell_coord(p.position, self.particle_effect_radius))
            .collect();

        self.current_particles.par_iter().enumerate().for_each(|(i, _)| {
            let index = hash(cell_coords[i]) % hash_table_length;
            hash_table[index].fetch_add(1, Relaxed);
        });

        let mut running_sum = 0;
        for i in 0..hash_table.len() - 1 {
            let current = hash_table[i].load(Relaxed);
            hash_table[i].store(running_sum, Relaxed);
            running_sum += current;
        }
        hash_table[hash_table_length].store(running_sum, Relaxed);

        let mut particle_indices = vec![0; self.current_particles.len()];
        
        let count_per_cell: Vec<_> = (0..hash_table_length)
            .map(|_| AtomicUsize::new(0))
            .collect();

        self.current_particles.iter().enumerate().for_each(|(i, p)| {
            let cell = cell_coord(p.position, self.particle_effect_radius);
            let hash_idx = hash(cell) % hash_table_length;
            let offset = count_per_cell[hash_idx].fetch_add(1, Relaxed);
            let index = hash_table[hash_idx].load(Relaxed) + offset;
            particle_indices[index] = i;
        });

        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();
        
        self.current_particles.reserve(self.previous_particles.len());

        let half_world = self.world_size * 0.5;
        let radius_squared = self.particle_effect_radius * self.particle_effect_radius;
        let scaled_force = self.force_scale * self.particle_effect_radius * ts;
        let friction_ts = self.friction * ts;

        self.current_particles.par_extend(
            self.previous_particles.par_iter().enumerate().map(|(particle_idx, _)| {
                let mut particle = self.previous_particles[particle_idx];
                let mut total_force = cgmath::Vector3::zero();
                
                let particle_cell = cell_coord(particle.position, self.particle_effect_radius);
                
                for x_offset in -1..=1 {
                    for y_offset in -1..=1 {
                        for z_offset in -1..=1 {
                            let world_offset = cgmath::vec3(
                                x_offset as f32, 
                                y_offset as f32, 
                                z_offset as f32
                            ) * self.world_size;
                            
                            for dx in -1..=1 {
                                for dy in -1..=1 {
                                    for dz in -1..=1 {
                                        let neighbor_cell = particle_cell + cgmath::vec3(dx, dy, dz);
                                        let cell_hash = hash(neighbor_cell) % hash_table_length;
                                        
                                        let start_idx = hash_table[cell_hash].load(Relaxed);
                                        let end_idx = hash_table[cell_hash + 1].load(Relaxed);
                                        
                                        if start_idx == end_idx {
                                            continue;
                                        }
                                        
                                        for idx in start_idx..end_idx {
                                            let other_idx = particle_indices[idx];
                                            let other_particle = &self.previous_particles[other_idx];
                                            
                                            let relative_pos = other_particle.position - (particle.position + world_offset);
                                            let sqr_distance = relative_pos.magnitude2();
                                            
                                            // skipped if too far or same particle
                                            if sqr_distance <= 0.0 || sqr_distance >= radius_squared {
                                                continue;
                                            }
                                            
                                            let distance = sqr_distance.sqrt();
                                            let attraction = self.attraction_matrix[(particle.id * self.id_count + other_particle.id) as usize];
                                            let f = calculate_force(distance / self.particle_effect_radius, attraction);
                                            
                                            total_force += relative_pos * (f / distance);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                particle.velocity += total_force * scaled_force;
                
                let velocity_magnitude = particle.velocity.magnitude2();
                let friction_magnitude = friction_ts * friction_ts * velocity_magnitude;
                
                if friction_magnitude >= velocity_magnitude {
                    particle.velocity = cgmath::Vector3::zero();
                } else {
                    let slowdown = 1.0 - friction_ts;
                    particle.velocity *= slowdown;
                }
                
                particle.position += particle.velocity * ts;
                
                if particle.position.x > half_world {
                    particle.position.x -= self.world_size;
                } else if particle.position.x < -half_world {
                    particle.position.x += self.world_size;
                }
                
                if particle.position.y > half_world {
                    particle.position.y -= self.world_size;
                } else if particle.position.y < -half_world {
                    particle.position.y += self.world_size;
                }
                
                if particle.position.z > half_world {
                    particle.position.z -= self.world_size;
                } else if particle.position.z < -half_world {
                    particle.position.z += self.world_size;
                }
                
                particle
            })
        );
    }
}