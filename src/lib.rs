use cgmath::prelude::*;
use encase::ShaderType;

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

        std::mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();
        
        // Sequential version - for each particle, check against all other particles
        for &particle in &self.previous_particles {
            let mut updated_particle = particle;
            let mut force = cgmath::Vector3::zero();
            
            //checking interactions with all other particles (O(n²) complexity)
            for &other_particle in &self.previous_particles {
                if other_particle.id == particle.id {
                    continue;
                }
                
                let offsets = [
                    cgmath::vec3(0.0, 0.0, 0.0),
                    cgmath::vec3(self.world_size, 0.0, 0.0),
                    cgmath::vec3(-self.world_size, 0.0, 0.0),
                    cgmath::vec3(0.0, self.world_size, 0.0),
                    cgmath::vec3(0.0, -self.world_size, 0.0),
                    cgmath::vec3(0.0, 0.0, self.world_size),
                    cgmath::vec3(0.0, 0.0, -self.world_size),
                    cgmath::vec3(self.world_size, self.world_size, 0.0),
                    cgmath::vec3(-self.world_size, self.world_size, 0.0),
                    cgmath::vec3(self.world_size, -self.world_size, 0.0),
                    cgmath::vec3(-self.world_size, -self.world_size, 0.0),
                    cgmath::vec3(self.world_size, 0.0, self.world_size),
                    cgmath::vec3(-self.world_size, 0.0, self.world_size),
                    cgmath::vec3(self.world_size, 0.0, -self.world_size),
                    cgmath::vec3(-self.world_size, 0.0, -self.world_size),
                    cgmath::vec3(0.0, self.world_size, self.world_size),
                    cgmath::vec3(0.0, -self.world_size, self.world_size),
                    cgmath::vec3(0.0, self.world_size, -self.world_size),
                    cgmath::vec3(0.0, -self.world_size, -self.world_size),
                    cgmath::vec3(self.world_size, self.world_size, self.world_size),
                    cgmath::vec3(-self.world_size, self.world_size, self.world_size),
                    cgmath::vec3(self.world_size, -self.world_size, self.world_size),
                    cgmath::vec3(self.world_size, self.world_size, -self.world_size),
                    cgmath::vec3(-self.world_size, -self.world_size, self.world_size),
                    cgmath::vec3(-self.world_size, self.world_size, -self.world_size),
                    cgmath::vec3(self.world_size, -self.world_size, -self.world_size),
                    cgmath::vec3(-self.world_size, -self.world_size, -self.world_size),
                ];
                
                for offset in &offsets {
                    let relative_position = other_particle.position - (updated_particle.position + *offset);
                    let sqr_distance = relative_position.magnitude2();
                    
                    if sqr_distance > 0.0 
                        && sqr_distance < self.particle_effect_radius * self.particle_effect_radius {
                        let distance = sqr_distance.sqrt();
                        let attraction_idx = (updated_particle.id * self.id_count + other_particle.id) as usize;
                        let f = self.calculate_force(
                            distance / self.particle_effect_radius,
                            self.attraction_matrix[attraction_idx],
                        );
                        force += relative_position / distance * f;
                    }
                }
            }
            
            updated_particle.velocity +=
                force * self.force_scale * self.particle_effect_radius * ts;
            updated_particle.velocity += self.gravity * ts;
            
            let velocity_change = updated_particle.velocity * self.friction * ts;
            if velocity_change.magnitude2() > updated_particle.velocity.magnitude2() {
                updated_particle.velocity = cgmath::vec3(0.0, 0.0, 0.0);
            } else {
                updated_particle.velocity -= velocity_change;
            }
            
            updated_particle.position += updated_particle.velocity * ts;
            self.handle_wall_collision(&mut updated_particle);
            
            self.current_particles.push(updated_particle);
        }
        
        self.current_particles.clone()
    }
}