use cgmath::prelude::*;
use eframe::egui_wgpu::wgpu;
use eframe::wgpu::include_wgsl;
use eframe::{egui, wgpu::util::DeviceExt};
use encase::{ArrayLength, ShaderSize, ShaderType, StorageBuffer, UniformBuffer};
use particle_life_3d::{Particle, Particles};
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

const ROTATION_SPEED: f32 = 90.0;
const MOVEMENT_SPEED: f32 = 5.0;

// Moved and reorganized structure definitions
#[derive(ShaderType)]
struct GpuCamera {
    pub view_matrix: cgmath::Matrix4<f32>,
    pub projection_matrix: cgmath::Matrix4<f32>,
}

struct Axes {
    pub forward: cgmath::Vector3<f32>,
    pub right: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
}

struct Camera {
    pub position: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub pitch: f32,
    pub yaw: f32,
}

#[derive(ShaderType)]
struct GpuParticles<'a> {
    pub world_size: f32,
    pub length: ArrayLength,
    #[size(runtime)]
    pub particles: &'a [Particle],
}

#[derive(ShaderType)]
struct GpuColors<'a> {
    pub length: ArrayLength,
    #[size(runtime)]
    pub particles: &'a [cgmath::Vector3<f32>],
}

// Camera implementation with renamed methods
impl Camera {
    fn calculate_axes(&self) -> Axes {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        
        let forward = cgmath::vec3(
            pitch_rad.cos() * yaw_rad.sin(),
            pitch_rad.sin(),
            -pitch_rad.cos() * yaw_rad.cos(),
        ).normalize();
        
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        
        Axes { forward, right, up }
    }
    
    fn new(position: cgmath::Vector3<f32>) -> Self {
        Self {
            position,
            up: cgmath::vec3(0.0, 1.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

// Restructured renderer implementation first
struct Renderer {
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    particles_storage_buffer: wgpu::Buffer,
    particles_storage_buffer_size: usize,
    colors_storage_buffer: wgpu::Buffer,
    colors_storage_buffer_size: usize,
    particles_bind_group_layout: wgpu::BindGroupLayout,
    particles_bind_group: wgpu::BindGroup,
    particles_render_pipeline: wgpu::RenderPipeline,
    border_render_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    fn new(render_state: &eframe::egui_wgpu::RenderState) -> Self {
        let device = &render_state.device;
        
        // Create shader modules in parallel
        let (particles_shader, border_shader) = rayon::join(
            || device.create_shader_module(include_wgsl!("./particles.wgsl")),
            || device.create_shader_module(include_wgsl!("./border.wgsl"))
        );

        let camera_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(<GpuCamera as ShaderSize>::SHADER_SIZE),
                    },
                    count: None,
                }],
            });

        let particles_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particles Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(<GpuParticles as ShaderType>::min_size()),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(<GpuColors as ShaderType>::min_size()),
                        },
                        count: None,
                    },
                ],
            });

        // Create buffers in parallel
        let (camera_uniform_buffer, particles_storage_buffer, colors_storage_buffer) = {
            let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: &[0; <GpuCamera as ShaderSize>::SHADER_SIZE.get() as _],
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });

            const PARTICLES_STORAGE_BUFFER_SIZE: usize =
                <GpuParticles as ShaderType>::METADATA.min_size().get() as _;
            const COLORS_STORAGE_BUFFER_SIZE: usize =
                <GpuColors as ShaderType>::METADATA.min_size().get() as _;
                
            let particles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particles Storage Buffer"),
                contents: &[0; PARTICLES_STORAGE_BUFFER_SIZE],
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            });

            let colors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Colors Storage Buffer"),
                contents: &[0; COLORS_STORAGE_BUFFER_SIZE],
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            });

            (camera_buffer, particles_buffer, colors_buffer)
        };

        // Create bind groups
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });

        let particles_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particles Bind Group"),
            layout: &particles_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: colors_storage_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipelines in parallel
        let (particles_render_pipeline, border_render_pipeline) = {
            let particles_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Particles Pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout, &particles_bind_group_layout],
                    push_constant_ranges: &[],
                });

            let border_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Border Pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout, &particles_bind_group_layout],
                    push_constant_ranges: &[],
                });

            rayon::join(
                || device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Particles Render Pipeline"),
                    layout: Some(&particles_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &particles_shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &particles_shader,
                        entry_point: "fs_main",
                        targets: &[Some(render_state.target_format.into())],
                    }),
                    primitive: wgpu::PrimitiveState {
                        polygon_mode: wgpu::PolygonMode::Fill,
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                }),
                || device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Border Render Pipeline"),
                    layout: Some(&border_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &border_shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &border_shader,
                        entry_point: "fs_main",
                        targets: &[Some(render_state.target_format.into())],
                    }),
                    primitive: wgpu::PrimitiveState {
                        polygon_mode: wgpu::PolygonMode::Line,
                        topology: wgpu::PrimitiveTopology::LineList,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                })
            )
        };

        Self {
            camera_uniform_buffer,
            camera_bind_group,
            particles_storage_buffer,
            particles_storage_buffer_size: <GpuParticles as ShaderType>::METADATA.min_size().get() as _,
            colors_storage_buffer,
            colors_storage_buffer_size: <GpuColors as ShaderType>::METADATA.min_size().get() as _,
            particles_bind_group_layout,
            particles_bind_group,
            particles_render_pipeline,
            border_render_pipeline,
        }
    }

    fn prepare(
        &mut self,
        camera: &[u8],
        particles: &[u8],
        colors: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &wgpu::CommandEncoder,
    ) -> Vec<wgpu::CommandBuffer> {
        // Instead of trying to clone the buffer, we'll use a reference
        let camera_buffer_ref = &self.camera_uniform_buffer;
        
        // Update camera - use queue directly instead of cloning
        queue.write_buffer(camera_buffer_ref, 0, camera);

        // Update particles and colors (potentially in parallel)
        let mut particles_bind_group_invalidated = false;
        
        if self.particles_storage_buffer_size >= particles.len() {
            queue.write_buffer(&self.particles_storage_buffer, 0, particles);
        } else {
            particles_bind_group_invalidated = true;
            self.particles_storage_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Particles Storage Buffer"),
                    contents: particles,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                });
            self.particles_storage_buffer_size = particles.len();
        }
        
        if self.colors_storage_buffer_size >= colors.len() {
            queue.write_buffer(&self.colors_storage_buffer, 0, colors);
        } else {
            particles_bind_group_invalidated = true;
            self.colors_storage_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Colors Storage Buffer"),
                    contents: colors,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                });
            self.colors_storage_buffer_size = colors.len();
        }
        
        if particles_bind_group_invalidated {
            self.particles_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particles Bind Group"),
                layout: &self.particles_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.colors_storage_buffer.as_entire_binding(),
                    },
                ],
            });
        }

        vec![]
    }

    fn paint<'a>(&'a self, sphere_count: u32, render_pass: &mut wgpu::RenderPass<'a>) {
        // First draw particles
        render_pass.set_pipeline(&self.particles_render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.particles_bind_group, &[]);
        render_pass.draw(0..4, 0..sphere_count);

        // Then draw border
        render_pass.set_pipeline(&self.border_render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.particles_bind_group, &[]);
        render_pass.draw(0..24, 0..1);
    }
}

// Main application structure
struct ParticleSimulation {
    particles: Particles,
    camera: Camera,
    last_frame_time: std::time::Instant,
    accumulated_time: std::time::Duration,
    simulation_rate: f32,
    particle_worker: Option<std::thread::JoinHandle<()>>,
    particles_shared: Arc<RwLock<Particles>>,
}

impl ParticleSimulation {
    fn new(cc: &eframe::CreationContext) -> Self {
        let particle_colors = vec![
            cgmath::vec3(1.0, 0.0, 0.0),   // red
            cgmath::vec3(0.0, 1.0, 0.0),   // green
            cgmath::vec3(0.0, 0.0, 1.0),   // blue
            cgmath::vec3(1.0, 1.0, 0.0),   // yellow
            cgmath::vec3(1.0, 0.0, 1.0),   // purple
        ];
        
        let particle_attraction = vec![
            0.5, 1.0, -0.5, 0.0, -1.0,   // red
            1.0, 1.0, 1.0, 0.0, -1.0,    // green
            0.0, 0.0, 0.5, 1.5, -1.0,    // blue
            0.0, 0.0, 0.0, 0.0, -1.0,    // yellow
            1.0, 1.0, 1.0, 1.0, 0.5,     // purple
        ];
        
        let world_size = 10.0;
        let particles = Particles {
            world_size,
            id_count: 5,
            colors: particle_colors,
            attraction_matrix: particle_attraction,
            particle_effect_radius: 2.0,
            friction: 0.97,
            force_scale: 1.0,
            current_particles: vec![],
            previous_particles: vec![],
        };

        // Initialize particles in parallel
        let rng = thread_rng();
        let current_particles: Vec<Particle> = (0..1000)
            .into_par_iter()
            .map(|_| {
                let mut thread_rng = thread_rng();
                Particle {
                    position: cgmath::vec3(
                        thread_rng.gen_range(world_size * -0.5..=world_size * 0.5),
                        thread_rng.gen_range(world_size * -0.5..=world_size * 0.5),
                        thread_rng.gen_range(world_size * -0.5..=world_size * 0.5),
                    ),
                    velocity: cgmath::vec3(0.0, 0.0, 0.0),
                    id: thread_rng.gen_range(0..5),
                }
            })
            .collect();
            
        let mut particles_instance = particles;
        particles_instance.current_particles = current_particles;

        // Setup renderer
        let render_state = cc.wgpu_render_state.as_ref().unwrap();
        let renderer = Renderer::new(render_state);
        render_state
            .renderer
            .write()
            .paint_callback_resources
            .insert(renderer);

        // Instead of cloning particles_instance, let's create a deep copy manually
        // assuming Particles has implemented the necessary traits or can be copied manually
        // Create a new Arc<RwLock<Particles>> with a new Particles instance
        let particles_for_shared = Particles {
            world_size: particles_instance.world_size,
            id_count: particles_instance.id_count,
            colors: particles_instance.colors.clone(),
            attraction_matrix: particles_instance.attraction_matrix.clone(),
            particle_effect_radius: particles_instance.particle_effect_radius,
            friction: particles_instance.friction,
            force_scale: particles_instance.force_scale,
            current_particles: particles_instance.current_particles.clone(),
            previous_particles: particles_instance.previous_particles.clone(),
        };
        
        let shared_particles = Arc::new(RwLock::new(particles_for_shared));

        ParticleSimulation {
            particles: particles_instance,
            camera: Camera::new(cgmath::vec3(1.0, 0.0, world_size * 1.6)),
            last_frame_time: std::time::Instant::now(),
            accumulated_time: std::time::Duration::ZERO,
            simulation_rate: 60.0,
            particle_worker: None,
            particles_shared: shared_particles,
        }
    }
    
    fn handle_camera_input(&mut self, ctx: &eframe::egui::Context, ts: f32) {
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                let axes = self.camera.calculate_axes();

                // Movement controls
                if i.key_down(egui::Key::W) {
                    self.camera.position += axes.forward * MOVEMENT_SPEED * ts;
                }
                if i.key_down(egui::Key::S) {
                    self.camera.position -= axes.forward * MOVEMENT_SPEED * ts;
                }
                if i.key_down(egui::Key::A) {
                    self.camera.position -= axes.right * MOVEMENT_SPEED * ts;
                }
                if i.key_down(egui::Key::D) {
                    self.camera.position += axes.right * MOVEMENT_SPEED * ts;
                }
                if i.key_down(egui::Key::Q) {
                    self.camera.position -= axes.up * MOVEMENT_SPEED * ts;
                }
                if i.key_down(egui::Key::E) {
                    self.camera.position += axes.up * MOVEMENT_SPEED * ts;
                }

                // Rotation controls
                if i.key_down(egui::Key::ArrowUp) {
                    self.camera.pitch += ROTATION_SPEED * ts;
                }
                if i.key_down(egui::Key::ArrowDown) {
                    self.camera.pitch -= ROTATION_SPEED * ts;
                }
                if i.key_down(egui::Key::ArrowLeft) {
                    self.camera.yaw -= ROTATION_SPEED * ts;
                }
                if i.key_down(egui::Key::ArrowRight) {
                    self.camera.yaw += ROTATION_SPEED * ts;
                }

                // Clamp pitch to prevent gimbal lock
                self.camera.pitch = self.camera.pitch.clamp(-89.9999, 89.9999);
            });
        }
    }
    
    fn render_ui_panel(&mut self, ui: &mut egui::Ui, frame_time: f32, update_time: f64) {
        ui.label(format!("FPS: {:.3}", 1.0 / frame_time));
        ui.label(format!("Frame Time: {:.3}ms", frame_time * 1000.0));
        ui.label(format!("Update Time: {:.3}ms", update_time * 1000.0));
        
        ui.horizontal(|ui| {
            ui.label("World Size: ");
            ui.add(egui::DragValue::new(&mut self.particles.world_size).speed(0.1));
            self.particles.world_size = self
                .particles
                .world_size
                .max(self.particles.particle_effect_radius * 2.0);
        });
        
        ui.horizontal(|ui| {
            ui.label("Simulation Rate: ");
            ui.add(egui::Slider::new(&mut self.simulation_rate, 1.0..=1000.0));
        });
        
        ui.horizontal(|ui| {
            ui.label("Friction: ");
            ui.add(
                egui::Slider::new(&mut self.particles.friction, 0.0..=1.0)
                    .drag_value_speed(0.01),
            );
        });
        
        ui.horizontal(|ui| {
            ui.label("Force Scale: ");
            ui.add(egui::Slider::new(
                &mut self.particles.force_scale,
                0.0..=10.0,
            ));
        });
    }
}

impl eframe::App for ParticleSimulation {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // Calculate time step
        let current_time = std::time::Instant::now();
        let frame_time = current_time.duration_since(self.last_frame_time);
        self.last_frame_time = current_time;
        
        let ts = frame_time.as_secs_f32();
        self.accumulated_time += frame_time;

        // Handle physics update in potentially parallel fashion
        let start_update = std::time::Instant::now();
        if self.accumulated_time.as_secs_f32() >= 1.0 / self.simulation_rate {
            let step_time = 1.0 / self.simulation_rate;
            
            // Option 1: Update in main thread
            self.particles.update(step_time);
            
            // Update shared particles for background thread work
            if let Ok(mut shared) = self.particles_shared.try_write() {
                // Instead of cloning, copy the fields individually
                shared.world_size = self.particles.world_size;
                shared.id_count = self.particles.id_count;
                shared.colors = self.particles.colors.clone();
                shared.attraction_matrix = self.particles.attraction_matrix.clone();
                shared.particle_effect_radius = self.particles.particle_effect_radius;
                shared.friction = self.particles.friction;
                shared.force_scale = self.particles.force_scale;
                shared.current_particles = self.particles.current_particles.clone();
                shared.previous_particles = self.particles.previous_particles.clone();
            }
            
            self.accumulated_time -= std::time::Duration::from_secs_f32(step_time);
        }
        let update_elapsed = start_update.elapsed();
        
        // Handle camera input
        self.handle_camera_input(ctx, ts);

        // Draw UI
        egui::SidePanel::left("Control Panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_ui_panel(ui, ts, update_elapsed.as_secs_f64());
                ui.allocate_space(ui.available_size());
            });
        });

        // Render particles
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(ctx.style().visuals.panel_fill))
            .show(ctx, |ui| {
                let (rect, _response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

                // Set up camera uniform
                let mut camera_uniform =
                    UniformBuffer::new([0; <GpuCamera as ShaderSize>::SHADER_SIZE.get() as _]);
                
                let axes = self.camera.calculate_axes();
                camera_uniform
                    .write(&GpuCamera {
                        view_matrix: cgmath::Matrix4::look_to_rh(
                            cgmath::point3(
                                self.camera.position.x,
                                self.camera.position.y,
                                self.camera.position.z,
                            ),
                            axes.forward,
                            axes.up,
                        ),
                        projection_matrix: cgmath::perspective(
                            cgmath::Rad::from(cgmath::Deg(90.0)),
                            rect.width() / rect.height(),
                            0.001,
                            1000.0,
                        ),
                    })
                    .unwrap();
                let camera = camera_uniform.into_inner();

                // Prepare particles data
                let (particles, colors) = rayon::join(
                    || {
                        let mut particles_storage = StorageBuffer::new(vec![]);
                        particles_storage
                            .write(&GpuParticles {
                                world_size: self.particles.world_size,
                                length: ArrayLength,
                                particles: &self.particles.current_particles,
                            })
                            .unwrap();
                        particles_storage.into_inner()
                    },
                    || {
                        let mut colors_storage = StorageBuffer::new(vec![]);
                        colors_storage
                            .write(&GpuColors {
                                length: ArrayLength,
                                particles: &self.particles.colors,
                            })
                            .unwrap();
                        colors_storage.into_inner()
                    }
                );

                let sphere_count = self.particles.current_particles.len();

                // Add paint callback
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: std::sync::Arc::new(
                        eframe::egui_wgpu::CallbackFn::new()
                            .prepare(move |device, queue, encoder, paint_callback_resources| {
                                let renderer: &mut Renderer =
                                    paint_callback_resources.get_mut().unwrap();
                                renderer.prepare(&camera, &particles, &colors, device, queue, encoder)
                            })
                            .paint(move |_info, render_pass, paint_callback_resources| {
                                let renderer: &Renderer = paint_callback_resources.get().unwrap();
                                renderer.paint(sphere_count as _, render_pass);
                            }),
                    ),
                });
            });

        // Request continuous updates
        ctx.request_repaint();
    }
}

fn main() {
    eframe::run_native(
        "Particle Physics 3D Simulator",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                present_mode: wgpu::PresentMode::AutoNoVsync,
                depth_format: Some(wgpu::TextureFormat::Depth32Float),
                device_descriptor: wgpu::DeviceDescriptor {
                    features: wgpu::Features::POLYGON_MODE_LINE,
                    ..Default::default()
                },
                ..Default::default()
            },
            vsync: false,
            depth_buffer: 32,
            ..Default::default()
        },
        Box::new(|cc| Box::new(ParticleSimulation::new(cc))),
    )
    .unwrap();
}