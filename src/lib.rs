use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use cgmath::prelude::*;
use encase::ShaderType;
use rayon::prelude::*;

//single particle with position, velocity, and identity
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct Particle {
    pub position: cgmath::Vector3<f32>,//where particle is in 3D space
    pub velocity: cgmath::Vector3<f32>,//how fast and which direction it's moving
    pub id: u32,//unique identifier for the particle
}

//entire particle system and its properties
pub struct Particles {
    pub world_size: f32,//size of the simulation box
    pub active_particles: Vec<Particle>,//current state of all particles
    pub past_particles: Vec<Particle>,//previous state (needed for calculations)
    pub id_count: u32,//total number of particle types
    pub attraction_matrix: Vec<f32>,//how much different particle types attract/repel each other
    pub colors: Vec<cgmath::Vector3<f32>>,//color for each particle type
    pub coefficient: f32,//how quickly particles slow down
    pub interaction_force: f32,//how strong the forces between particles are
    pub min_pull_ratio: f32,//minimum distance where attraction happens
    pub particle_effect_radius: f32,//how far particles can affect each other
    pub walls: bool, //whether particles bounce off walls or wrap around
    pub acceleration: cgmath::Vector3<f32>, //direction and strength of gravity
}

impl Particles {
    //checking out which grid cell a particle is in (for faster neighbor finding)
    fn cell_coord(&self, v: cgmath::Vector3<f32>) -> cgmath::Vector3<isize> {
        cgmath::vec3(
            (v.x / self.particle_effect_radius) as isize,
            (v.y / self.particle_effect_radius) as isize,
            (v.z / self.particle_effect_radius) as isize,
        )
    }

    //converting a 3D grid cell into a single number for the hash table
    fn hash_cell(cell: cgmath::Vector3<isize>) -> usize {
        let mut hasher = DefaultHasher::new();
        cell.x.hash(&mut hasher);
        cell.y.hash(&mut hasher);
        cell.z.hash(&mut hasher);
        hasher.finish() as usize
    }

    //checking how strongly particles interact based on distance and attraction value
    fn calculate_force(&self, distance: f32, attraction: f32) -> f32 {
        if distance < self.min_pull_ratio {
            //very close particles repel each other
            distance / self.min_pull_ratio - 1.0
        } else if self.min_pull_ratio < distance && distance < 1.0 {
            //medium distance particles attract or repel based on the attraction matrix
            attraction * (1.0 - (2.0 * distance - 1.0 - self.min_pull_ratio)
                .abs() / (1.0 - self.min_pull_ratio))
        } else {
            //far particles don't affect each other
            0.0
        }
    }

    //handling what happens when particles hit the world boundaries
    fn handle_wall_collision(&self, particle: &mut Particle) {
        let half_world = self.world_size * 0.5;
        
        //x-axis wall handling
        if particle.position.x > half_world {
            if self.walls {
                //bounce off wall
                particle.position.x = half_world;
                particle.velocity.x = particle.velocity.x.min(0.0);
            } else {
                //wrap around to other side
                particle.position.x -= self.world_size;
            }
        } else if particle.position.x < -half_world {
            if self.walls {
                //bounce off wall
                particle.position.x = -half_world;
                particle.velocity.x = particle.velocity.x.max(0.0);
            } else {
                //wrap around to other side
                particle.position.x += self.world_size;
            }
        }

        //y-axis wall handling 
        if particle.position.y > half_world {
            if self.walls {
                particle.position.y = half_world;
                particle.velocity.y = particle.velocity.y.min(0.0);
            } else {
                particle.position.y -= self.world_size;
            }
        } else if particle.position.y < -half_world {
            if self.walls {
                particle.position.y = -half_world;
                particle.velocity.y = particle.velocity.y.max(0.0);
            } else {
                particle.position.y += self.world_size;
            }
        }

        //z-axis wall handling 
        if particle.position.z > half_world {
            if self.walls {
                particle.position.z = half_world;
                particle.velocity.z = particle.velocity.z.min(0.0);
            } else {
                particle.position.z -= self.world_size;
            }
        } else if particle.position.z < -half_world {
            if self.walls {
                particle.position.z = -half_world;
                particle.velocity.z = particle.velocity.z.max(0.0);
            } else {
                particle.position.z += self.world_size;
            }
        }
    }

    //updating all particles for one time step
    pub fn update(&mut self, ts: f32) -> Vec<Particle> {
        //making sure the world is big enough for our particle effects
        assert!(self.world_size >= 2.0 * self.particle_effect_radius);

        //setting up a spatial hash table to quickly find nearby particles
        let hash_table_length = self.active_particles.len();
        let hash_table: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(hash_table_length + 1)
            .collect();

        //parallely counting how many particles are in each grid cell
        self.active_particles.par_iter().for_each(|sphere| {
            let index = Self::hash_cell(self.cell_coord(sphere.position)) % hash_table_length;
            hash_table[index].fetch_add(1, Relaxed);
        });

        //converting counts to running totals to create index ranges
        for i in 1..hash_table.len() {
            hash_table[i].fetch_add(hash_table[i - 1].load(Relaxed), Relaxed);
        }

        //creating array to store which particle is in which position
        let particle_indices: Vec<_> = std::iter::repeat_with(|| AtomicUsize::new(0))
            .take(self.active_particles.len())
            .collect();

        //filling the particle indices array parallely
        self.active_particles
            .par_iter()
            .enumerate()
            .for_each(|(i, sphere)| {
                let index = Self::hash_cell(self.cell_coord(sphere.position)) % hash_table_length;
                let index = hash_table[index].fetch_sub(1, Relaxed);
                particle_indices[index - 1].store(i, Relaxed);
            });

        //swaping current and previous particle arrays and prepare for update
        std::mem::swap(&mut self.active_particles, &mut self.past_particles);
        self.active_particles.clear();
        
        //processing each particle in parallel
        self.active_particles = self.past_particles
            .par_iter()
            .map(|&particle| {
                let mut updated_particle = particle;
                
                //parallel calculating total force on this particle from all nearby particles
                let total_force = (-1..=1)
                    .into_par_iter()
                    .flat_map(|x_offset| {
                        (-1..=1).into_par_iter().flat_map(move |y_offset| {
                            (-1..=1).into_par_iter().map(move |z_offset| {
                                (x_offset, y_offset, z_offset)
                            })
                        })
                    })
                    .fold(
                        || cgmath::Vector3::zero(), 
                        |mut acc, (x_offset, y_offset, z_offset)| {
                            //handling particles that might be on the other side of boundary
                            let offset = cgmath::vec3(x_offset as _, y_offset as _, z_offset as _)
                                * self.world_size;
                            let cell = self.cell_coord(updated_particle.position + offset);

                            //checking all neighboring cells for nearby particles
                            for x_cell_offset in -1..=1 {
                                for y_cell_offset in -1..=1 {
                                    for z_cell_offset in -1..=1 {
                                        let cell = cell
                                            + cgmath::vec3(x_cell_offset, y_cell_offset, z_cell_offset);

                                        //looking up particles in this cell using our hash table
                                        let index = Self::hash_cell(cell) % hash_table_length;
                                        for index in &particle_indices[hash_table[index]
                                            .load(Relaxed)
                                            ..hash_table[index + 1].load(Relaxed)]
                                        {
                                            let other_particle =
                                                &self.past_particles[index.load(Relaxed)];

                                            //calculating distance to the other particle
                                            let relative_position = other_particle.position
                                                - (updated_particle.position + offset);
                                            let sqr_distance = relative_position.magnitude2();
                                            
                                            //if it is close enough to affect each other and not the same particle
                                            if sqr_distance > 0.0
                                                && sqr_distance
                                                    < self.particle_effect_radius
                                                        * self.particle_effect_radius
                                            {
                                                let distance = sqr_distance.sqrt();
                                                //get force from attraction matrix based on particle types
                                                let f = self.calculate_force(
                                                    distance,
                                                    self.attraction_matrix[(updated_particle.id
                                                        * self.id_count
                                                        + other_particle.id)
                                                        as usize],
                                                );
                                                //adding force vector to accumulated force
                                                acc += relative_position / distance * f;
                                            }
                                        }
                                    }
                                }
                            }
                            acc
                        }
                    )
                    .reduce(
                        || cgmath::Vector3::zero(), 
                        |a, b| a + b
                    );

                //updating velocity based on calculated forces
                updated_particle.velocity +=
                    total_force * self.interaction_force * self.particle_effect_radius * ts;
                //applying gravity
                updated_particle.velocity += self.acceleration * ts;

                //applying friction to slow particles down
                let velocity_change = updated_particle.velocity * self.coefficient * ts;
                if velocity_change.magnitude2() > updated_particle.velocity.magnitude2() {
                    //stopping completely if friction would reverse direction
                    updated_particle.velocity = cgmath::vec3(0.0, 0.0, 0.0);
                } else {
                    //otherwise just slow down
                    updated_particle.velocity -= velocity_change;
                }

                //updating position based on velocity
                updated_particle.position += updated_particle.velocity * ts;
                //handling collisions with world boundaries
                self.handle_wall_collision(&mut updated_particle);

                updated_particle
            })
            .collect();

        //returning the updated particles
        self.active_particles.clone()
    }
}