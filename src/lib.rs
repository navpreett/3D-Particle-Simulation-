use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use cgmath::prelude::*;
use encase::ShaderType;
use rayon::prelude::*;

#[derive(Clone, Copy, ShaderType, Debug)]
pub struct Particle {
    pub position: cgmath::Vector3<f32>,
    pub velocity: cgmath::Vector3<f32>,
    pub id: u32,
}

pub struct Particles {
    pub world_size: f32,
    pub current_particles: Vec<Particle>,
    pub previous_particles: Vec<Particle>,
    pub id_count: u32,
    pub attraction_matrix: Vec<f32>,
    pub colors: Vec<cgmath::Vector3<f32>>,
    pub friction: f32,
    pub force_scale: f32,
    pub min_attraction_percentage: f32,
    pub particle_effect_radius: f32,
    pub solid_walls: bool,
    pub gravity: cgmath::Vector3<f32>,
}

impl Particles {
    fn cell_coord(&self, v: cgmath::Vector3<f32>) -> cgmath::Vector3<isize> {
        cgmath::vec3(
            (v.x / self.particle_effect_radius) as isize,
            (v.y / self.particle_effect_radius) as isize,
            (v.z / self.particle_effect_radius) as isize,
        )
    }

    fn hash_cell(cell: cgmath::Vector3<isize>) -> usize {
        let mut hasher = DefaultHasher::new();
        cell.x.hash(&mut hasher);
        cell.y.hash(&mut hasher);
        cell.z.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn calculate_force(&self, distance: f32, attraction: f32) -> f32 {
        if distance < self.min_attraction_percentage {
            distance / self.min_attraction_percentage - 1.0
        } else if self.min_attraction_percentage < distance && distance < 1.0 {
            attraction * (1.0 - (2.0 * distance - 1.0 - self.min_attraction_percentage)
                .abs() / (1.0 - self.min_attraction_percentage))
        } else {
            0.0
        }
    }

    fn handle_wall_collision(&self, particle: &mut Particle) {
        let half_world = self.world_size * 0.5;
        
        if particle.position.x > half_world {
            if self.solid_walls {
                particle.position.x = half_world;
                particle.velocity.x = particle.velocity.x.min(0.0);
            } else {
                particle.position.x -= self.world_size;
            }
        } else if particle.position.x < -half_world {
            if self.solid_walls {
                particle.position.x = -half_world;
                particle.velocity.x = particle.velocity.x.max(0.0);
            } else {
                particle.position.x += self.world_size;
            }
        }

        if particle.position.y > half_world {
            if self.solid_walls {
                particle.position.y = half_world;
                particle.velocity.y = particle.velocity.y.min(0.0);
            } else {
                particle.position.y -= self.world_size;
            }
        } else if particle.position.y < -half_world {
            if self.solid_walls {
                particle.position.y = -half_world;
                particle.velocity.y = particle.velocity.y.max(0.0);
            } else {
                particle.position.y += self.world_size;
            }
        }

        if particle.position.z > half_world {
            if self.solid_walls {
                particle.position.z = half_world;
                particle.velocity.z = particle.velocity.z.min(0.0);
            } else {
                particle.position.z -= self.world_size;
            }
        } else if particle.position.z < -half_world {
            if self.solid_walls {
                particle.position.z = -half_world;
                particle.velocity.z = particle.velocity.z.max(0.0);
            } else {
                particle.position.z += self.world_size;
            }
        }
    }

    pub fn update(&mut self, ts: f32) -> Vec<Particle> {
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);

        let hash_table_length = self.current_particles.len();
        let hash_table: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(hash_table_length + 1)
            .collect();

        self.current_particles.par_iter().for_each(|sphere| {
            let index = Self::hash_cell(self.cell_coord(sphere.position)) % hash_table_length;
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
                let index = Self::hash_cell(self.cell_coord(sphere.position)) % hash_table_length;
                let index = hash_table[index].fetch_sub(1, Relaxed);
                particle_indices[index - 1].store(i, Relaxed);
            });

        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();
        
        let offsets: Vec<_> = (-1..=1)
            .flat_map(|x_offset| {
                (-1..=1).flat_map(move |y_offset| {
                    (-1..=1).map(move |z_offset| {
                        (x_offset, y_offset, z_offset)
                    })
                })
            })
            .collect();

        self.current_particles = self.previous_particles
            .par_iter()
            .map(|&particle| {
                let mut updated_particle = particle;
                
                let total_force = offsets
                    .par_iter()
                    .map(|&(x_offset, y_offset, z_offset)| {
                        let mut force = cgmath::Vector3::zero();
                        let offset = cgmath::vec3(x_offset as f32, y_offset as f32, z_offset as f32)
                            * self.world_size;
                        let cell = self.cell_coord(updated_particle.position + offset);

                        for x_cell_offset in -1..=1 {
                            for y_cell_offset in -1..=1 {
                                for z_cell_offset in -1..=1 {
                                    let neighbor_cell = cell
                                        + cgmath::vec3(x_cell_offset, y_cell_offset, z_cell_offset);

                                    let cell_hash = Self::hash_cell(neighbor_cell) % hash_table_length;
                                    let start = hash_table[cell_hash].load(Relaxed);
                                    let end = hash_table[cell_hash + 1].load(Relaxed);
                                    
                                    for index in start..end {
                                        let other_particle_idx = particle_indices[index].load(Relaxed);
                                        let other_particle = &self.previous_particles[other_particle_idx];

                                        let relative_position = other_particle.position
                                            - (updated_particle.position + offset);
                                        let sqr_distance = relative_position.magnitude2();
                                        
                                        if sqr_distance > 0.0
                                            && sqr_distance
                                                < self.particle_effect_radius
                                                    * self.particle_effect_radius
                                        {
                                            let distance = sqr_distance.sqrt();
                                            let f = self.calculate_force(
                                                distance / self.particle_effect_radius,
                                                self.attraction_matrix[(updated_particle.id
                                                    * self.id_count
                                                    + other_particle.id)
                                                    as usize],
                                            );
                                            force += relative_position / distance * f;
                                        }
                                    }
                                }
                            }
                        }
                        force
                    })
                    .reduce(
                        || cgmath::Vector3::zero(), 
                        |a, b| a + b
                    );

                updated_particle.velocity +=
                    total_force * self.force_scale * self.particle_effect_radius * ts;
                updated_particle.velocity += self.gravity * ts;

                let velocity_change = updated_particle.velocity * self.friction * ts;
                if velocity_change.magnitude2() > updated_particle.velocity.magnitude2() {
                    updated_particle.velocity = cgmath::vec3(0.0, 0.0, 0.0);
                } else {
                    updated_particle.velocity -= velocity_change;
                }

                updated_particle.position += updated_particle.velocity * ts;
                self.handle_wall_collision(&mut updated_particle);

                updated_particle
            })
            .collect();

        self.current_particles.clone()
    }
}