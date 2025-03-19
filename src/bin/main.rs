use cgmath::prelude::*;
use eframe::egui;
use particle_life_3d::{Particle, Particles};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const TIME_STEP: f32 = 1.0 / 60.0;
const CAMERA_SPEED: f32 = 5.0;
const CAMERA_ROTATION_SPEED: f32 = 90.0;

#[derive(Clone, Copy)]
struct Axes {
    pub forward: cgmath::Vector3<f32>,
    pub right: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
}

#[derive(Clone, Copy)]
struct Camera {
    pub position: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub pitch: f32,
    pub yaw: f32,
}

impl Camera {
    fn new(position: cgmath::Vector3<f32>) -> Self {
        Camera {
            position,
            up: cgmath::vec3(0.0, 1.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
        }
    }
    
    pub fn get_axes(&self) -> Axes {
        let forward = cgmath::vec3(
            self.pitch.to_radians().cos() * (-self.yaw).to_radians().sin(),
            self.pitch.to_radians().sin(),
            self.pitch.to_radians().cos() * (-self.yaw).to_radians().cos(),
        )
        .normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        Axes { forward, right, up }
    }
    
    pub fn move_direction(&mut self, direction: &cgmath::Vector3<f32>, axes: &Axes, delta_time: f32) {
        if direction.x != 0.0 {
            self.position += axes.right * direction.x * CAMERA_SPEED * delta_time;
        }
        if direction.y != 0.0 {
            self.position += axes.up * direction.y * CAMERA_SPEED * delta_time;
        }
        if direction.z != 0.0 {
            self.position += axes.forward * direction.z * CAMERA_SPEED * delta_time;
        }
    }
    
    pub fn rotate(&mut self, x_rot: f32, y_rot: f32, delta_time: f32) {
        self.yaw += x_rot * CAMERA_ROTATION_SPEED * delta_time;
        self.pitch += y_rot * CAMERA_ROTATION_SPEED * delta_time;
        self.pitch = self.pitch.clamp(-89.9999, 89.9999);
    }
}

struct App {
    particles: Arc<Mutex<Particles>>,
    camera: Camera,
    last_time: std::time::Instant,
    fixed_time: std::time::Duration,
    frame_stats: FrameStats,
}

struct FrameStats {
    fps: f32,
    frame_time_ms: f32,
    update_time_ms: f64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_ms: 0.0,
            update_time_ms: 0.0,
        }
    }
}

impl App {
    fn new(_cc: &eframe::CreationContext) -> Self {
        let particles = Self::create_particles();
        let world_size = particles.lock().unwrap().world_size;
        
        let camera = Camera::new(cgmath::vec3(0.0, 0.0, -world_size * 1.5));

        Self {
            particles,
            camera,
            last_time: std::time::Instant::now(),
            fixed_time: std::time::Duration::ZERO,
            frame_stats: FrameStats::default(),
        }
    }
    
    fn create_particles() -> Arc<Mutex<Particles>> {
        let mut particles = Particles {
            world_size: 10.0,
            id_count: 5,
            colors: vec![
                cgmath::vec3(1.0, 0.0, 0.0), // red
                cgmath::vec3(0.0, 1.0, 0.0), // green
                cgmath::vec3(0.0, 0.0, 1.0), // blue
                cgmath::vec3(1.0, 1.0, 0.0), // yellow
                cgmath::vec3(1.0, 0.0, 1.0), // purple
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
        
        let world_size = particles.world_size;
        
        let particle_data: Vec<(cgmath::Vector3<f32>, usize)> = (0..1000)
            .into_par_iter()
            .map(|_| {
                let mut local_rng = thread_rng();
                let pos = cgmath::vec3(
                    local_rng.gen_range(world_size * -0.5..=world_size * 0.5),
                    local_rng.gen_range(world_size * -0.5..=world_size * 0.5),
                    local_rng.gen_range(world_size * -0.5..=world_size * 0.5),
                );
                let id = local_rng.gen_range(0..5);
                (pos, id)
            })
            .collect();
            
        particles.current_particles = particle_data
            .into_iter()
            .map(|(position, id)| Particle {
                position,
                velocity: cgmath::vec3(0.0, 0.0, 0.0),
                id,
            })
            .collect();

        Arc::new(Mutex::new(particles))
    }
    
    fn handle_input(&mut self, ctx: &egui::Context, response: &egui::Response, delta_time: f32) {
        if !response.has_focus() {
            return;
        }
        
        ctx.input(|i| {
            let axes = self.camera.get_axes();
            
            let mut movement = cgmath::vec3(0.0, 0.0, 0.0);
            if i.key_pressed(egui::Key::W) { movement.z += 1.0; }
            if i.key_pressed(egui::Key::S) { movement.z -= 1.0; }
            if i.key_pressed(egui::Key::A) { movement.x -= 1.0; }
            if i.key_pressed(egui::Key::D) { movement.x += 1.0; }
            if i.key_pressed(egui::Key::Q) { movement.y -= 1.0; }
            if i.key_pressed(egui::Key::E) { movement.y += 1.0; }
            
            if movement.x != 0.0 || movement.y != 0.0 || movement.z != 0.0 {
                self.camera.move_direction(&movement, &axes, delta_time);
            }
            
            let mut x_rot = 0.0;
            let mut y_rot = 0.0;
            if i.key_pressed(egui::Key::ArrowLeft) { x_rot -= 1.0; }
            if i.key_pressed(egui::Key::ArrowRight) { x_rot += 1.0; }
            if i.key_pressed(egui::Key::ArrowUp) { y_rot += 1.0; }
            if i.key_pressed(egui::Key::ArrowDown) { y_rot -= 1.0; }
            
            if x_rot != 0.0 || y_rot != 0.0 {
                self.camera.rotate(x_rot, y_rot, delta_time);
            }
        });
    }
    
    fn update_physics(&mut self) -> std::time::Duration {
        let start_update = std::time::Instant::now();
        
        if self.fixed_time.as_secs_f32() >= TIME_STEP {
            let particles_arc = Arc::clone(&self.particles);
            
            std::thread::spawn(move || {
                let mut particles = particles_arc.lock().unwrap();
                particles.update(TIME_STEP);
            }).join().unwrap();
            
            self.fixed_time -= std::time::Duration::from_secs_f32(TIME_STEP);
        }
        
        start_update.elapsed()
    }
    
    fn render_ui(&self, ctx: &egui::Context) -> egui::Response {
        egui::SidePanel::left("Left Panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(format!("FPS: {:.3}", self.frame_stats.fps));
                ui.label(format!("Frame Time: {:.3}ms", self.frame_stats.frame_time_ms));
                ui.label(format!("Update Time: {:.3}ms", self.frame_stats.update_time_ms));
                ui.allocate_space(ui.available_size());
            });
        });

        egui::CentralPanel::default()
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(egui::Vec2::splat(300.0), egui::Sense::drag());
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: std::sync::Arc::new(
                        eframe::egui_wgpu::CallbackFn::new()
                            .prepare(
                                move |_device, _queue, _encoder, _paint_callback_resources| {
                                    Vec::new()
                                },
                            )
                            .paint(move |_info, _render_pass, _paint_callback_resources| {}),
                    ),
                });

                response
            })
            .inner
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let time = std::time::Instant::now();
        let time_delta = time.duration_since(self.last_time);
        let delta_time = time_delta.as_secs_f32();
        self.last_time = time;
        
        self.frame_stats.fps = 1.0 / delta_time;
        self.frame_stats.frame_time_ms = delta_time * 1000.0;
        
        self.fixed_time += time_delta;
        let update_elapsed = self.update_physics();
        self.frame_stats.update_time_ms = update_elapsed.as_secs_f64() * 1000.0;
        
        let response = self.render_ui(ctx);
        self.handle_input(ctx, &response, delta_time);
        
        ctx.request_repaint();
    }
}

fn main() {
    eframe::run_native(
        "Particle Physics 3D",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            vsync: false,
            ..Default::default()
        },
        Box::new(|cc| Box::new(App::new(cc))),
    )
    .unwrap();
}