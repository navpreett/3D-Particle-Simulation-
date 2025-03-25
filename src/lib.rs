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

#[derive(Clone)]
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
}


impl Particles {
    fn compute_cell_coord(&self, v: cgmath::Vector3<f32>) -> cgmath::Vector3<isize> {
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

    fn compute_particle_force(
        &self,
        particle: &Particle,
        other_particle: &Particle,
        distance: f32,
    ) -> cgmath::Vector3<f32> {
        let attraction_value = self.attraction_matrix[(particle.id * self.id_count + other_particle.id) as usize];
        
        let force = |dist: f32, attraction: f32| -> f32 {
            if dist < self.min_attraction_percentage {
                dist / self.min_attraction_percentage - 1.0
            } else if self.min_attraction_percentage < dist && dist < 1.0 {
                attraction * (1.0 - (2.0 * dist - 1.0 - self.min_attraction_percentage).abs() 
                    / (1.0 - self.min_attraction_percentage))
            } else {
                0.0
            }
        };

        let f = force(distance, attraction_value);
        let relative_position = other_particle.position - particle.position;
        relative_position / distance * f
    }

    fn update_particle_kinematics(&self, particle: &mut Particle, total_force: cgmath::Vector3<f32>, ts: f32) {
        particle.velocity += total_force * self.force_scale * self.particle_effect_radius * ts;
        let velocity_change = particle.velocity * self.friction * ts;
        
        if velocity_change.magnitude2() > particle.velocity.magnitude2() {
            particle.velocity = cgmath::vec3(0.0, 0.0, 0.0);
        } else {
            particle.velocity -= velocity_change;
        }

        particle.position += particle.velocity * ts;
        particle.position.x = self.wrap_coordinate(particle.position.x);
        particle.position.y = self.wrap_coordinate(particle.position.y);
        particle.position.z = self.wrap_coordinate(particle.position.z);
    }

    fn wrap_coordinate(&self, coord: f32) -> f32 {
        let half_world_size = self.world_size * 0.5;
        if coord > half_world_size {
            coord - self.world_size
        } else if coord < -half_world_size {
            coord + self.world_size
        } else {
            coord
        }
    }

    pub fn update(&mut self, ts: f32) {
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);

        let hash_table_length = self.current_particles.len();
        let hash_table: Vec<AtomicUsize> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(hash_table_length + 1)
            .collect();

        self.current_particles.par_iter().for_each(|sphere| {
            let index = Self::hash_cell(self.compute_cell_coord(sphere.position)) % hash_table_length;
            hash_table[index].fetch_add(1, Relaxed);
        });

        for i in 1..hash_table.len() {
            hash_table[i].fetch_add(hash_table[i - 1].load(Relaxed), Relaxed);
        }

        let particle_indices: Vec<AtomicUsize> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(self.current_particles.len())
            .collect();

        self.current_particles
            .par_iter()
            .enumerate()
            .for_each(|(i, sphere)| {
                let index = Self::hash_cell(self.compute_cell_coord(sphere.position)) % hash_table_length;
                let index = hash_table[index].fetch_sub(1, Relaxed);
                particle_indices[index - 1].store(i, Relaxed);
            });

        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();
        
        self.current_particles = self.previous_particles
            .par_iter()
            .map(|particle| {
                let mut updated_particle = *particle;
                let mut total_force = cgmath::Vector3::zero();

                for x_offset in -1..=1 {
                    for y_offset in -1..=1 {
                        for z_offset in -1..=1 {
                            let offset = cgmath::vec3(x_offset as _, y_offset as _, z_offset as _)
                                * self.world_size;
                            
                            let nearby_particles = self.find_nearby_particles(
                                particle, 
                                offset, 
                                &particle_indices, 
                                &hash_table, 
                                hash_table_length
                            );

                            total_force += nearby_particles
                                .iter()
                                .map(|&other_index| {
                                    let other_particle = &self.previous_particles[other_index];
                                    let relative_pos = other_particle.position - (particle.position + offset);
                                    let sqr_distance = relative_pos.magnitude2();
                                    
                                    if sqr_distance > 0.0 && 
                                       sqr_distance < self.particle_effect_radius * self.particle_effect_radius {
                                        let distance = sqr_distance.sqrt();
                                        self.compute_particle_force(particle, other_particle, distance)
                                    } else {
                                        cgmath::Vector3::zero()
                                    }
                                })
                                .sum();
                        }
                    }
                }

                self.update_particle_kinematics(&mut updated_particle, total_force, ts);
                updated_particle
            })
            .collect();
    }

    fn find_nearby_particles(
        &self,
        particle: &Particle,
        offset: cgmath::Vector3<f32>,
        particle_indices: &[AtomicUsize],
        hash_table: &[AtomicUsize],
        hash_table_length: usize,
    ) -> Vec<usize> {
        let mut nearby_particles = Vec::new();
        let cell = self.compute_cell_coord(particle.position + offset);

        for x_cell_offset in -1isize..=1 {
            for y_cell_offset in -1isize..=1 {
                for z_cell_offset in -1isize..=1 {
                    let target_cell = cell + cgmath::vec3(x_cell_offset, y_cell_offset, z_cell_offset);
                    let index = Self::hash_cell(target_cell) % hash_table_length;
                    
                    nearby_particles.extend(
                        (hash_table[index].load(Relaxed)..hash_table[index + 1].load(Relaxed))
                        .map(|idx| particle_indices[idx].load(Relaxed))
                    );
                }
            }
        }

        nearby_particles
    }
}