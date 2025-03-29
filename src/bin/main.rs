use std::cmp::Ordering;

use cgmath::prelude::*;
use eframe::egui_wgpu::wgpu;
use eframe::wgpu::include_wgsl;
use eframe::{egui, wgpu::util::DeviceExt};
use encase::{ArrayLength, ShaderSize, ShaderType, StorageBuffer, UniformBuffer};
use particle_3d::{Particle, Particles};
use rand::prelude::*;
use rayon::prelude::*;

const CAMERA_SPEED: f32 = 5.0;
const CAMERA_ROTATION_SPEED: f32 = 90.0;
const MAX_PARTICLE_TYPES: usize = 5;

#[derive(Clone)]
struct CameraSystem {
    position: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
    pitch: f32,
    yaw: f32,
}

impl CameraSystem {
    fn calculate_axes(&self) -> (cgmath::Vector3<f32>, cgmath::Vector3<f32>, cgmath::Vector3<f32>) {
        let pitch_rad = self.pitch.to_radians();
        let yaw_rad = self.yaw.to_radians();
        
        let pitch_cos = pitch_rad.cos();
        let pitch_sin = pitch_rad.sin();
        let yaw_sin = yaw_rad.sin();
        let yaw_cos = yaw_rad.cos();
        
        let forward = cgmath::vec3(
            pitch_cos * yaw_sin,
            pitch_sin,
            -pitch_cos * yaw_cos,
        )
        .normalize();
        
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        
        (forward, right, up)
    }

    fn move_camera(&mut self, delta: f32, input_vector: cgmath::Vector3<f32>) {
        self.position += input_vector * CAMERA_SPEED * delta;
    }

    fn rotate_camera(&mut self, pitch_delta: f32, yaw_delta: f32) {
        self.pitch += pitch_delta;
        self.yaw += yaw_delta;
        self.pitch = self.pitch.clamp(-89.9999, 89.9999);
    }
}

fn generate_particles(world_size: f32, count: usize) -> Vec<Particle> {
    (0..count)
        .into_par_iter()
        .map_init(
            || rand::thread_rng(),
            |rng, _| {
                let half_size = world_size * 0.5;
                Particle {
                    position: cgmath::vec3(
                        rng.gen_range(-half_size..=half_size),
                        rng.gen_range(-half_size..=half_size),
                        rng.gen_range(-half_size..=half_size),
                    ),
                    velocity: cgmath::vec3(0.0, 0.0, 0.0),
                    id: rng.gen_range(0..MAX_PARTICLE_TYPES) as u32,
                }
            },
        )
        .collect()
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

#[derive(ShaderType)]
struct GpuCamera {
    pub view_matrix: cgmath::Matrix4<f32>,
    pub projection_matrix: cgmath::Matrix4<f32>,
}

struct SimulationApp {
    particles: Particles,
    camera: CameraSystem,
    last_time: std::time::Instant,
    fixed_time: std::time::Duration,
    ticks_per_second: f32,
    color_window_open: bool,
}

impl SimulationApp {
    fn new(cc: &eframe::CreationContext) -> Self {
        let particles = Particles {
            world_size: 10.0,
            id_count: MAX_PARTICLE_TYPES as u32,
            colors: vec![
                cgmath::vec3(1.0, 0.0, 0.0),
                cgmath::vec3(0.0, 1.0, 0.0),
                cgmath::vec3(0.0, 0.0, 1.0),
                cgmath::vec3(1.0, 1.0, 0.0),
                cgmath::vec3(1.0, 0.0, 1.0),
            ],
            attraction_matrix: vec![
                0.5, 1.0, -0.5, 0.0, -1.0,
                1.0, 1.0, 1.0, 0.0, -1.0,
                0.0, 0.0, 0.5, 1.5, -1.0,
                0.0, 0.0, 0.0, 0.0, -1.0,
                1.0, 1.0, 1.0, 1.0, 0.5,
            ],
            particle_effect_radius: 2.0,
            friction: 0.97,
            force_scale: 1.0,
            min_attraction_percentage: 0.3,
            current_particles: generate_particles(10.0, 1000),
            previous_particles: vec![],
            solid_walls: false,
            gravity: cgmath::vec3(0.0, 0.0, 0.0),
        };

        let camera = CameraSystem {
            position: cgmath::vec3(1.0, 0.0, particles.world_size * 1.6),
            up: cgmath::vec3(0.0, 1.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
        };

        let app = Self {
            particles,
            camera,
            last_time: std::time::Instant::now(),
            fixed_time: std::time::Duration::ZERO,
            ticks_per_second: 60.0,
            color_window_open: false,
        };

        let render_state = cc.wgpu_render_state.as_ref().unwrap();
        let renderer = Renderer::new(render_state);
        render_state
            .renderer
            .write()
            .paint_callback_resources
            .insert(renderer);

        app
    }
}

impl eframe::App for SimulationApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        let time = std::time::Instant::now();
        let ts = time.duration_since(self.last_time);
        self.last_time = time;

        self.fixed_time += ts;
        let start_update = std::time::Instant::now();
        if self.fixed_time.as_secs_f32() >= 1.0 / self.ticks_per_second {
            let ts = 1.0 / self.ticks_per_second;
            let fixed_step = std::time::Duration::from_secs_f32(1.0 / self.ticks_per_second);
            
            let updates_needed = (self.fixed_time.as_secs_f32() * self.ticks_per_second).min(5.0) as usize;
            for _ in 0..updates_needed {
                self.particles.update(ts);
                self.fixed_time -= fixed_step;
            }
        }
        let update_elapsed = start_update.elapsed();

        let ts = ts.as_secs_f32();

        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                let (forward, right, up) = self.camera.calculate_axes();

                if i.key_down(egui::Key::W) {
                    self.camera.move_camera(ts, forward);
                }
                if i.key_down(egui::Key::S) {
                    self.camera.move_camera(ts, -forward);
                }
                if i.key_down(egui::Key::A) {
                    self.camera.move_camera(ts, -right);
                }
                if i.key_down(egui::Key::D) {
                    self.camera.move_camera(ts, right);
                }
                if i.key_down(egui::Key::Q) {
                    self.camera.move_camera(ts, -up);
                }
                if i.key_down(egui::Key::E) {
                    self.camera.move_camera(ts, up);
                }

                if i.key_down(egui::Key::ArrowUp) {
                    self.camera.rotate_camera(CAMERA_ROTATION_SPEED * ts, 0.0);
                }
                if i.key_down(egui::Key::ArrowDown) {
                    self.camera.rotate_camera(-CAMERA_ROTATION_SPEED * ts, 0.0);
                }
                if i.key_down(egui::Key::ArrowLeft) {
                    self.camera.rotate_camera(0.0, -CAMERA_ROTATION_SPEED * ts);
                }
                if i.key_down(egui::Key::ArrowRight) {
                    self.camera.rotate_camera(0.0, CAMERA_ROTATION_SPEED * ts);
                }
            });
        }

        egui::SidePanel::left("Left Panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(format!("FPS: {:.3}", 1.0 / ts));
                ui.label(format!("Frame Time: {:.3}ms", ts * 1000.0));
                ui.label(format!(
                    "Update Time: {:.3}ms",
                    update_elapsed.as_secs_f64() * 1000.0
                ));
                ui.horizontal(|ui| {
                    ui.label("Particle Count: ");
                    let mut particle_count = self.particles.current_particles.len();
                    if ui
                        .add(egui::DragValue::new(&mut particle_count).speed(0.1))
                        .changed()
                    {
                        match particle_count.cmp(&self.particles.current_particles.len()) {
                            Ordering::Less => {
                                self.particles.current_particles.truncate(particle_count);
                            }
                            Ordering::Greater => {
                                let additional = particle_count - self.particles.current_particles.len();
                                self.particles.current_particles.reserve(additional);
                                
                                let new_particles = generate_particles(self.particles.world_size, additional);
                                self.particles.current_particles.extend(new_particles);
                            }
                            Ordering::Equal => {}
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("World Size: ");
                    ui.add(egui::DragValue::new(&mut self.particles.world_size).speed(0.1));
                    self.particles.world_size = self
                        .particles
                        .world_size
                        .max(self.particles.particle_effect_radius * 2.0);
                });
                ui.horizontal(|ui| {
                    ui.label("Solid Walls: ");
                    ui.checkbox(&mut self.particles.solid_walls, "");
                });
                ui.horizontal(|ui| {
                    ui.label("Ticks Per Second: ");
                    ui.add(egui::Slider::new(&mut self.ticks_per_second, 1.0..=1000.0));
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
                ui.horizontal(|ui| {
                    ui.label("Particle Effect Radius: ");
                    ui.add(egui::Slider::new(
                        &mut self.particles.particle_effect_radius,
                        0.0..=self.particles.world_size / 2.0,
                    ));
                });
                ui.horizontal(|ui| {
                    ui.label("Repulsion Distance Percentage: ");
                    ui.add(egui::Slider::new(
                        &mut self.particles.min_attraction_percentage,
                        0.0..=1.0,
                    ));
                });
                ui.horizontal(|ui| {
                    ui.label("Gravity: ");
                    ui.add(
                        egui::DragValue::new(&mut self.particles.gravity.x)
                            .prefix("x: ")
                            .speed(0.01),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.particles.gravity.y)
                            .prefix("y: ")
                            .speed(0.01),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.particles.gravity.z)
                            .prefix("z: ")
                            .speed(0.01),
                    );
                });
                self.color_window_open |= ui.button("Particle Properties").clicked();
                ui.allocate_space(ui.available_size());
            });
        });

        egui::Window::new("Properties")
            .open(&mut self.color_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let size = ui.spacing().interact_size;
                    ui.allocate_exact_size(size, egui::Sense::hover());

                    for i in 0..self.particles.id_count {
                        let mut ui_color = [
                            self.particles.colors[i as usize].x,
                            self.particles.colors[i as usize].y,
                            self.particles.colors[i as usize].z,
                        ];
                        ui.color_edit_button_rgb(&mut ui_color);
                        self.particles.colors[i as usize] =
                            cgmath::vec3(ui_color[0], ui_color[1], ui_color[2]);
                    }
                });
                for i in 0..self.particles.id_count {
                    ui.horizontal(|ui| {
                        let mut ui_color = [
                            self.particles.colors[i as usize].x,
                            self.particles.colors[i as usize].y,
                            self.particles.colors[i as usize].z,
                        ];
                        ui.color_edit_button_rgb(&mut ui_color);
                        self.particles.colors[i as usize] =
                            cgmath::vec3(ui_color[0], ui_color[1], ui_color[2]);

                        for j in 0..self.particles.id_count {
                            ui.add(
                                egui::DragValue::new(
                                    &mut self.particles.attraction_matrix
                                        [(i * self.particles.id_count + j) as usize],
                                )
                                .clamp_range(-1.0..=1.0)
                                .speed(0.01),
                            );
                        }
                    });
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(ctx.style().visuals.panel_fill))
            .show(ctx, |ui| {
                let (rect, _response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

                let mut camera_uniform =
                    UniformBuffer::new([0; <GpuCamera as ShaderSize>::SHADER_SIZE.get() as _]);
                camera_uniform
                    .write(&{
                        let (forward, _, up) = self.camera.calculate_axes();
                        GpuCamera {
                            view_matrix: cgmath::Matrix4::look_to_rh(
                                cgmath::point3(
                                    self.camera.position.x,
                                    self.camera.position.y,
                                    self.camera.position.z,
                                ),
                                forward,
                                up,
                            ),
                            projection_matrix: cgmath::perspective(
                                cgmath::Rad::from(cgmath::Deg(90.0)),
                                rect.width() / rect.height(),
                                0.001,
                                1000.0,
                            ),
                        }
                    })
                    .unwrap();
                let camera = camera_uniform.into_inner();

                let mut particles_storage = StorageBuffer::new(vec![]);
                particles_storage
                    .write(&GpuParticles {
                        world_size: self.particles.world_size,
                        length: ArrayLength,
                        particles: &self.particles.current_particles,
                    })
                    .unwrap();
                let particles = particles_storage.into_inner();

                let mut colors_storage = StorageBuffer::new(vec![]);
                colors_storage
                    .write(&GpuColors {
                        length: ArrayLength,
                        particles: &self.particles.colors,
                    })
                    .unwrap();
                let colors = colors_storage.into_inner();

                let sphere_count = self.particles.current_particles.len();

                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: std::sync::Arc::new(
                        eframe::egui_wgpu::CallbackFn::new()
                            .prepare(move |device, queue, encoder, paint_callback_resources| {
                                let renderer: &mut Renderer =
                                    paint_callback_resources.get_mut().unwrap();
                                renderer
                                    .prepare(&camera, &particles, &colors, device, queue, encoder)
                            })
                            .paint(move |_info, render_pass, paint_callback_resources| {
                                let renderer: &Renderer = paint_callback_resources.get().unwrap();
                                renderer.paint(sphere_count as _, render_pass);
                            }),
                    ),
                });

                ctx.request_repaint();
            });
    }
}

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
        let particles_shader = render_state
            .device
            .create_shader_module(include_wgsl!("./particles.wgsl"));

        let border_shader = render_state
            .device
            .create_shader_module(include_wgsl!("./border.wgsl"));

        let camera_bind_group_layout =
            render_state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let camera_uniform_buffer =
            render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Uniform Buffer"),
                    contents: &[0; <GpuCamera as ShaderSize>::SHADER_SIZE.get() as _],
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                });

        let camera_bind_group = render_state
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Camera Bind Group"),
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buffer.as_entire_binding(),
                }],
            });

        let particles_bind_group_layout =
            render_state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        const PARTICLES_STORAGE_BUFFER_SIZE: usize =
            <GpuParticles as ShaderType>::METADATA.min_size().get() as _;
        let particles_storage_buffer =
            render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Particles Storage Buffer"),
                    contents: &[0; PARTICLES_STORAGE_BUFFER_SIZE],
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                });

        const COLORS_STORAGE_BUFFER_SIZE: usize =
            <GpuColors as ShaderType>::METADATA.min_size().get() as _;
        let colors_storage_buffer =
            render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Particles Storage Buffer"),
                    contents: &[0; COLORS_STORAGE_BUFFER_SIZE],
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                });

        let particles_bind_group =
            render_state
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
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

        let particles_pipeline_layout =
            render_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Particles Pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout, &particles_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let particles_render_pipeline =
            render_state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    multisample: wgpu::MultisampleState {
                        ..Default::default()
                    },
                    multiview: None,
                });

        let border_pipeline_layout =
            render_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Border Pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout, &particles_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let border_render_pipeline =
            render_state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    multisample: wgpu::MultisampleState {
                        ..Default::default()
                    },
                    multiview: None,
                });

        Self {
            camera_uniform_buffer,
            camera_bind_group,
            particles_storage_buffer,
            particles_storage_buffer_size: PARTICLES_STORAGE_BUFFER_SIZE,
            colors_storage_buffer,
            colors_storage_buffer_size: COLORS_STORAGE_BUFFER_SIZE,
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
        queue.write_buffer(&self.camera_uniform_buffer, 0, camera);
        
        let mut particles_bind_group_invalidated = false;
        
        let aligned_particles_len = (particles.len() + 3) & !3; 
        if self.particles_storage_buffer_size < aligned_particles_len {
            let new_size = ((aligned_particles_len as f32 * 1.2) as usize + 3) & !3;
            particles_bind_group_invalidated = true;
            
            self.particles_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particles Storage Buffer"),
                size: new_size as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            
            self.particles_storage_buffer_size = new_size;
        }
        
        let aligned_colors_len = (colors.len() + 3) & !3;
        if self.colors_storage_buffer_size < aligned_colors_len {
            let new_size = ((aligned_colors_len as f32 * 1.2) as usize + 3) & !3;
            particles_bind_group_invalidated = true;
            
            self.colors_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Colors Storage Buffer"),
                size: new_size as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            
            self.colors_storage_buffer_size = new_size;
        }
        
        queue.write_buffer(&self.particles_storage_buffer, 0, particles);
        queue.write_buffer(&self.colors_storage_buffer, 0, colors);
        
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
        render_pass.set_pipeline(&self.particles_render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.particles_bind_group, &[]);
        render_pass.draw(0..4, 0..sphere_count);

        render_pass.set_pipeline(&self.border_render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.particles_bind_group, &[]);
        render_pass.draw(0..24, 0..1);
    }
}

fn main() {
    eframe::run_native(
        "3D Particle",
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
        Box::new(|cc| Box::new(SimulationApp::new(cc))),
    )
    .unwrap();
}