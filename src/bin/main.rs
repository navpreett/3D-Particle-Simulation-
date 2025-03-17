use ::rand::prelude::*;
use macroquad::prelude::*;
use particle_life_3d::{Particle, Particles};
#[derive(Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub id: usize,
}
const CAMERA_SPEED: f32 = 5.0;
const CAMERA_ROTATION_SPEED: f32 = 90.0;
const TIME_STEP: f32 = 1.0 / 60.0;
const WORLD_SIZE: f32 = 10.0;
const PARTICLE_COUNT: usize = 1000;
const PARTICLE_RADIUS: f32 = 0.05;

fn config() -> Conf {
    Conf {
        window_title: "3D Particle Simulation".into(),
        icon: None,
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
struct Camera {
    position: Vec3,
    up: Vec3,
    pitch: f32,
    yaw: f32,
}

impl Camera {
    fn new() -> Self {
        Camera {
            position: vec3(0.0, 0.0, -WORLD_SIZE * 1.5),
            up: vec3(0.0, 1.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
        }
    }

    fn get_axes(&self) -> (Vec3, Vec3, Vec3) {
        let forward = vec3(
            self.pitch.to_radians().cos() * (-self.yaw).to_radians().sin(),
            self.pitch.to_radians().sin(),
            self.pitch.to_radians().cos() * (-self.yaw).to_radians().cos(),
        )
        .normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        (forward, right, up)
    }

    fn update(&mut self, delta_time: f32) {
        let (forward, right, up) = self.get_axes();

        if is_key_down(KeyCode::W) {
            self.position += forward * CAMERA_SPEED * delta_time;
        }
        if is_key_down(KeyCode::S) {
            self.position -= forward * CAMERA_SPEED * delta_time;
        }
        if is_key_down(KeyCode::A) {
            self.position -= right * CAMERA_SPEED * delta_time;
        }
        if is_key_down(KeyCode::D) {
            self.position += right * CAMERA_SPEED * delta_time;
        }
        if is_key_down(KeyCode::Q) {
            self.position -= up * CAMERA_SPEED * delta_time;
        }
        if is_key_down(KeyCode::E) {
            self.position += up * CAMERA_SPEED * delta_time;
        }

        if is_key_down(KeyCode::Up) {
            self.pitch += CAMERA_ROTATION_SPEED * delta_time;
        }
        if is_key_down(KeyCode::Down) {
            self.pitch -= CAMERA_ROTATION_SPEED * delta_time;
        }
        if is_key_down(KeyCode::Left) {
            self.yaw -= CAMERA_ROTATION_SPEED * delta_time;
        }
        if is_key_down(KeyCode::Right) {
            self.yaw += CAMERA_ROTATION_SPEED * delta_time;
        }

        self.pitch = self.pitch.clamp(-89.9999, 89.9999);
    }

    fn to_camera3d(&self) -> Camera3D {
        let (forward, _, _) = self.get_axes();
        Camera3D {
            position: self.position,
            target: self.position + forward,
            up: self.up,
            ..Default::default()
        }
    }
}

fn initialize_particles() -> Particles {
    let mut particles = Particles {
        world_size: WORLD_SIZE,
        id_count: 5,
        colors: vec![
            Color::from_vec(vec4(1.0, 0.0, 0.0, 1.0)), // red
            Color::from_vec(vec4(0.0, 1.0, 0.0, 1.0)), // green
            Color::from_vec(vec4(0.0, 0.0, 1.0, 1.0)), // blue
            Color::from_vec(vec4(1.0, 1.0, 0.0, 1.0)), // yellow
            Color::from_vec(vec4(1.0, 0.0, 1.0, 1.0)), // purple
        ],
        attraction_matrix: vec![
            0.5, 1.0, -0.5, 0.0, -1.0, // red
            1.0, 1.0, 1.0, 0.0, -1.0, // green
            0.0, 0.0, 0.5, 1.5, -1.0, // blue
            0.0, 0.0, 0.0, 0.0, -1.0, // yellow
            1.0, 1.0, 1.0, 1.0, 0.5, // purple
        ],
        particle_effect_radius: 2.0,
        friction_half_time: 0.04,
        force_scale: 1.0,
        current_particles: vec![],
        previous_particles: vec![],
    };

    let mut rng = thread_rng();
    particles.current_particles = (0..PARTICLE_COUNT)
        .map(|_| Particle {
            position: vec3(
                rng.gen_range(-WORLD_SIZE / 2.0..=WORLD_SIZE / 2.0),
                rng.gen_range(-WORLD_SIZE / 2.0..=WORLD_SIZE / 2.0),
                rng.gen_range(-WORLD_SIZE / 2.0..=WORLD_SIZE / 2.0),
            ),
            velocity: vec3(0.0, 0.0, 0.0),
            id: rng.gen_range(0..5),
        })
        .collect();

    particles
}

fn draw_particles(particles: &Particles) {
    for particle in &particles.current_particles {
        draw_sphere(particle.position, PARTICLE_RADIUS, None, particles.colors[particle.id]);
    }
}

fn draw_debug_info(frame_time: f32, update_time: std::time::Duration) {
    draw_text(&format!("FPS: {:.3}", 1.0 / frame_time), 5.0, 16.0, 16.0, WHITE);
    draw_text(
        &format!("Frame Time: {:.3}ms", frame_time * 1000.0),
        5.0,
        32.0,
        16.0,
        WHITE,
    );
    draw_text(
        &format!("Update Time: {:.3}ms", update_time.as_secs_f64() * 1000.0),
        5.0,
        48.0,
        16.0,
        WHITE,
    );
}

#[macroquad::main(config)]
async fn main() {
    let mut particles = initialize_particles();
    let mut camera = Camera::new();
    let mut fixed_time = 0.0;

    set_cursor_grab(true);
    show_mouse(false);

    loop {
        let frame_time = get_frame_time();
        fixed_time += frame_time;

        let start_update = std::time::Instant::now();
        if fixed_time >= TIME_STEP {
            particles.update(frame_time);
            fixed_time -= TIME_STEP;
        }
        let update_elapsed = start_update.elapsed();

        camera.update(frame_time);

        clear_background(Color::from_vec(vec4(0.1, 0.1, 0.1, 1.0)));
        set_camera(&camera.to_camera3d());

        draw_cube_wires(
            vec3(0.0, 0.0, 0.0),
            vec3(WORLD_SIZE, WORLD_SIZE, WORLD_SIZE),
            Color::from_vec(vec4(1.0, 1.0, 1.0, 1.0)),
        );

        draw_particles(&particles);

        set_default_camera();
        draw_debug_info(frame_time, update_elapsed);

        next_frame().await;
    }
}