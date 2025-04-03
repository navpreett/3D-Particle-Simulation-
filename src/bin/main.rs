use cgmath::prelude::*;
use encase::{ArrayLength, ShaderSize, ShaderType, StorageBuffer, UniformBuffer};
use eframe::{egui, wgpu::util::DeviceExt};
use particle_3d::{Particle, Particles};
use eframe::egui_wgpu::wgpu;
use eframe::wgpu::include_wgsl;
use rand::prelude::*;
use rayon::prelude::*;

//constants for movement and particle types
const ROTATION_SPEED: f32 = 90.0;
const SPEED: f32 = 5.0;
const MAX_PARTICLE_TYPES: usize = 5;

//camera system to control position, direction, and movements
#[derive(Clone)]
struct CameraSystem {
    position: cgmath::Vector3<f32>, //3d camera in space
    up: cgmath::Vector3<f32>,
    pitch: f32, //up/down rotation
    yaw: f32, //left/right rotation
}

impl CameraSystem {
        //calculation for the camera's forward, right, and up direction vectors
    fn calculate_axes(&self) -> (cgmath::Vector3<f32>, cgmath::Vector3<f32>, cgmath::Vector3<f32>) {
        let pitch_rad = self.pitch.to_radians();
        let yaw_rad = self.yaw.to_radians();
        
        let pitch_cos = pitch_rad.cos();
        let pitch_sin = pitch_rad.sin();
        let yaw_sin = yaw_rad.sin();
        let yaw_cos = yaw_rad.cos();
        // Calculation for the forward direction linked with pitch and yaw
        let forward = cgmath::vec3(
            pitch_cos * yaw_sin,
            pitch_sin,
            -pitch_cos * yaw_cos,
        )
        .normalize();
    //compute right and up vectors using cross products
    let right = forward.cross(self.up).normalize();
    let up = forward.cross(right).normalize();
    
    (forward, right, up)//3 direction vector
    }
    //movement of camera based on user input and time delta
    fn move_camera(&mut self, delta: f32, input_vector: cgmath::Vector3<f32>) {
        self.position += input_vector * SPEED * delta;
    }
    //rotates the camera by modifying pitch and yaw
    fn rotate_camera(&mut self, pitch_delta: f32, yaw_delta: f32) {
        self.pitch += pitch_delta; //pitch angle
        self.yaw += yaw_delta;
        self.pitch = self.pitch.clamp(-90.9999, 90.9999);//avoid flipping 
    }
}

//creating a range from 0 to count 
fn generate_particles(world_size: f32, count: usize) -> Vec<Particle> {
    (0..count)
        .into_par_iter()//speed up processing
        .map_init(
            || rand::thread_rng(),//creating a random number generator for each thread
            |rng, _| {
                let half_size = world_size * 0.5;//calculate half of world size for positioning
                let position = cgmath::Vector3::new(
                    rng.gen_range(-half_size..=half_size),//random X position
                    rng.gen_range(-half_size..=half_size),
                    rng.gen_range(-half_size..=half_size),
                );
                //starting with no movement
                let velocity = cgmath::Vector3::new(0.0, 0.0, 0.0); 
                //assigning a random type ID
                let id = rng.gen_range(0..MAX_PARTICLE_TYPES) as u32;
                
                Particle {//storing generated values
                    position,
                    velocity,
                    id,
                }
                
            },
        )
        .collect()//get all generated particles into a vector and return

}

#[derive(ShaderType)]
struct GpuParticles<'a> {
    //simulation size
    pub world_size: f32,
    pub length: ArrayLength,//active particles size
    #[size(runtime)]
    pub particles: &'a [Particle],//storing particle data in compatible with gpu
}

#[derive(ShaderType)]
struct GpuColors<'a> {
    pub length: ArrayLength,//no. of colors available for particles
    #[size(runtime)]
    pub particles: &'a [cgmath::Vector3<f32>],//storing color data for each particle
}

#[derive(ShaderType)]
struct GpuCamera {
    pub view_matrix: cgmath::Matrix4<f32>,//camera's view transformation
    pub projection_matrix: cgmath::Matrix4<f32>,//camera's projection transformation
}

struct SimulationApp {
    particles: Particles,//holding all particle data and behavior
    camera: CameraSystem,//handling the 3D camera view
    last_time: std::time::Instant, //tracking when the last frame was processed
    fixed_time: std::time::Duration,//accumulated time for physics updates
    update_rate: f32,//how many physics updates per second
    window: bool,//controls if settings window is shown
}

impl SimulationApp {
    fn new(cc: &eframe::CreationContext) -> Self {
        //creating a new particle system with initial settings
        let particles = Particles {
            world_size: 10.0, //size of the simulation space
            id_count: MAX_PARTICLE_TYPES as u32,//no. of different particle types
            colors: vec![//colors for different particle types
                cgmath::vec3(1.0, 0.0, 0.0), // red
                cgmath::vec3(0.0, 1.0, 0.0), // green
                cgmath::vec3(0.0, 0.0, 1.0), // blue
                cgmath::vec3(1.0, 1.0, 0.0), // yellow
                cgmath::vec3(1.0, 0.0, 1.0), // magenta
            ],
            attraction_matrix: vec![//how different particles attract/repel each other
                0.5, 1.0, -0.5, 0.0, -1.0,//positive values = attraction, negative = repulsion
                1.0, 1.0, 1.0, 0.0, -1.0,
                0.0, 0.0, 0.5, 1.5, -1.0,
                0.0, 0.0, 0.0, 0.0, -1.0,
                1.0, 1.0, 1.0, 1.0, 0.5,
            ],
            particle_effect_radius: 2.0,//how far particles can affect each other
            coefficient: 0.97,//friction drag (1.0 = no friction)
            interaction_force: 1.0,//strength of particle interactions
            min_pull_ratio: 0.3, //when to push instead of pull
            active_particles: generate_particles(10.0, 1000),//creating 1000 starting particles
            past_particles: vec![],//storage for previous frames (not used here)
            walls: false,//whether particles bounce off walls
            acceleration: cgmath::vec3(0.0, 0.0, 0.0),  // gravity
        };

        //setting up camera
        let camera = CameraSystem {
            position: cgmath::vec3(1.0, 0.0, particles.world_size * 1.6),//starting position
            up: cgmath::vec3(0.0, 1.0, 0.0),// way is up
            pitch: 0.0, //looking up/down angle
            yaw: 0.0,//looking left/right angle
        };

        //main app with everything initialized
        let app = Self {
            particles,
            camera,
            last_time: std::time::Instant::now(),//starting timing now
            fixed_time: std::time::Duration::ZERO,//no accumulated time yet
            update_rate: 60.0, //physics updates 60 times per second
            window: false,//start with settings window closed
        };

        //setting up the graphics renderer
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
        //calculating time since last frame
        let time = std::time::Instant::now();
        let ts = time.duration_since(self.last_time);
        self.last_time = time;

        //handling physics updates at a fixed rate (for stability)
        self.fixed_time += ts;
        let start_update = std::time::Instant::now();
        if self.fixed_time.as_secs_f32() >= 1.0 / self.update_rate {
            let ts = 1.0 / self.update_rate;
            let fixed_step = std::time::Duration::from_secs_f32(1.0 / self.update_rate);
            
            //catching up on physics if its behind, but not too many at once
            let updates_needed = (self.fixed_time.as_secs_f32() * self.update_rate).min(5.0) as usize;
            for _ in 0..updates_needed {
                self.particles.update(ts); //updating all particle positions
                self.fixed_time -= fixed_step;//subtracting the time i just simulated
            }
        }
        let update_elapsed = start_update.elapsed();//checking how long physics updates it took

        let ts = ts.as_secs_f32();//converting time to seconds for movement calculations

        //handling keyboard input for camera movement
        if !ctx.wants_keyboard_input() {    //won't move camera if typing in a text field
            ctx.input(|i| {
                //camera's current position
                let (forward, right, up) = self.camera.calculate_axes();

                //WASD keys for moving camera
                if i.key_down(egui::Key::W) {
                    self.camera.move_camera(ts, forward);// forward
                }
                if i.key_down(egui::Key::S) {
                    self.camera.move_camera(ts, -forward); // backward
                }
                if i.key_down(egui::Key::A) {
                    self.camera.move_camera(ts, -right); // left
                }
                if i.key_down(egui::Key::D) {
                    self.camera.move_camera(ts, right);// right
                }
                if i.key_down(egui::Key::Q) {
                    self.camera.move_camera(ts, -up);// down
                }
                if i.key_down(egui::Key::E) {
                    self.camera.move_camera(ts, up);// up
                }

                //arrow keys for rotating camera
                if i.key_down(egui::Key::ArrowUp) {
                    self.camera.rotate_camera(ROTATION_SPEED * ts, 0.0);// up
                }
                if i.key_down(egui::Key::ArrowDown) {
                    self.camera.rotate_camera(-ROTATION_SPEED * ts, 0.0);// down
                }
                if i.key_down(egui::Key::ArrowLeft) {
                    self.camera.rotate_camera(0.0, -ROTATION_SPEED * ts);//left
                }
                if i.key_down(egui::Key::ArrowRight) {
                    self.camera.rotate_camera(0.0, ROTATION_SPEED * ts); //right
                }
            });
        }

        //creating and filling the side panel with controls
        egui::SidePanel::left("Left Panel").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                //show performance information
                ui.label(format!("FPS: {:.3}", 1.0 / ts)); //frames per second
                ui.label(format!("Frame Time: {:.3}ms", ts * 1000.0));//time per frame
                ui.label(format!(
                    "Updated Time: {:.3}ms",
                    update_elapsed.as_secs_f64() * 1000.0//time for physics
                ));
                
                //slider to change number of particles
                ui.horizontal(|ui| {
                    ui.label("Particle Count: ");
                    let mut particle_count = self.particles.active_particles.len();
                    if ui
                        .add(egui::DragValue::new(&mut particle_count).speed(0.1))
                        .changed()
                    {
                        let current_count = self.particles.active_particles.len();
                        if particle_count < current_count {
                            //remove particles if I decreased the count
                            self.particles.active_particles.truncate(particle_count);
                        } else if particle_count > current_count {
                            //add new particles if I increased the count
                            let additional = particle_count - current_count;
                            self.particles.active_particles.reserve(additional);
                            let new_particles = generate_particles(self.particles.world_size, additional);
                            self.particles.active_particles.extend(new_particles);
                        }
                    }
                });
                
                //controlling for simulation boundary size
                ui.horizontal(|ui| {
                    ui.label("Simulation Boundary: ");
                    ui.add(egui::DragValue::new(&mut self.particles.world_size).speed(0.1));
                    //making sure the world is at least big enough for particle interactions
                    self.particles.world_size = self
                        .particles
                        .world_size
                        .max(self.particles.particle_effect_radius * 2.0);
                });
                
                //controlling for physics update rate
                ui.horizontal(|ui| {
                    ui.label("Update Rate (TPS): ");
                    ui.add(egui::Slider::new(&mut self.update_rate, 1.0..=1000.0));
                });
                
                //toggling for solid walls
                ui.horizontal(|ui| {
                    ui.label("Use Solid Walls: ");
                    ui.checkbox(&mut self.particles.walls, "");//checking to make particles bounce off walls
                });
                
                //controlling for how far particles can affect each other
                ui.horizontal(|ui| {
                    ui.label("Effect Radius: ");
                    ui.add(egui::Slider::new(
                        &mut self.particles.particle_effect_radius,
                        0.0..=self.particles.world_size / 2.0,
                    ));
                });
                
                //controlling for strength of particle interactions
                ui.horizontal(|ui| {
                    ui.label("Interaction Scale Rate: ");
                    ui.add(egui::Slider::new(
                        &mut self.particles.interaction_force,
                        0.0..=10.0,
                    ));
                });
                
                //toggling for friction
                ui.horizontal(|ui| {
                    ui.label("Drag (Friction): ");
                    ui.add(
                        egui::Slider::new(&mut self.particles.coefficient, 0.0..=1.0)
                            .drag_value_speed(0.01),
                    );
                });
                
                //controlling for when to push vs pull particles
                ui.horizontal(|ui| {
                    ui.label("Repulsion Threshold: ");
                    ui.add(egui::Slider::new(
                        &mut self.particles.min_pull_ratio,
                        0.0..=1.0,
                    ));
                });
                
                //toggling for gravity
                ui.horizontal(|ui| {
                    ui.label("Global Gravity: ");
                    ui.add(
                        egui::DragValue::new(&mut self.particles.acceleration.y)
                            .prefix("y: ")
                            .speed(0.01),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.particles.acceleration.x)
                            .prefix("x: ")
                            .speed(0.01),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.particles.acceleration.z)
                            .prefix("z: ")
                            .speed(0.01),
                    );
                });
                
                //button to open particle settings window
                self.window |= ui.button("Particle Settings").clicked();
                ui.allocate_space(ui.available_size());
            });
        });

        egui::Window::new("Properties")
        .open(&mut self.window)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                for i in 0..self.particles.id_count as usize {
                    let mut ui_color = [
                        self.particles.colors[i].x,
                        self.particles.colors[i].y,
                        self.particles.colors[i].z,
                    ];
                    ui.color_edit_button_rgb(&mut ui_color);
                    self.particles.colors[i] = cgmath::vec3(ui_color[0], ui_color[1], ui_color[2]);
                }
            });
            
            for i in 0..self.particles.id_count as usize {
                ui.horizontal(|ui| {
                    // Show color for this row
                    let mut ui_color = [
                        self.particles.colors[i].x,
                        self.particles.colors[i].y,
                        self.particles.colors[i].z,
                    ];
                    ui.color_edit_button_rgb(&mut ui_color);
                    self.particles.colors[i] = cgmath::vec3(ui_color[0], ui_color[1], ui_color[2]);
                    
                    //attraction/repulsion sliders for each particle type
                    for j in 0..self.particles.id_count as usize {
                        ui.add(
                            egui::DragValue::new(&mut self.particles.attraction_matrix[i * self.particles.id_count as usize + j])
                                .clamp_range(-1.0..=1.0)
                                .speed(0.01)
                        );
                    }
                });
            }
        });
        //created the main 3d view panel
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
                         //created view matrix (camera position and orientation)
                            view_matrix: cgmath::Matrix4::look_to_rh(
                                cgmath::point3(
                                    self.camera.position.x,
                                    self.camera.position.y,
                                    self.camera.position.z,
                                ),
                                forward,
                                up,
                            ),
                             //created projection matrix
                            projection_matrix: cgmath::perspective(
                                cgmath::Rad::from(cgmath::Deg(90.0)),//90 degree field of view
                                rect.width() / rect.height(),//screen ratio 
                                0.001,//clipping plane
                                1000.0,
                            ),
                        }
                    })
                    .unwrap();
                let camera = camera_uniform.into_inner();
                //preparing particle data for gpu
                let mut particles_storage = StorageBuffer::new(vec![]);
                particles_storage
                    .write(&GpuParticles {
                        world_size: self.particles.world_size,
                        length: ArrayLength,
                        particles: &self.particles.active_particles,
                    })
                    .unwrap();
                let particles = particles_storage.into_inner();
                //preparing color data for gpu
                let mut colors_storage = StorageBuffer::new(vec![]);
                colors_storage
                    .write(&GpuColors {
                        length: ArrayLength,
                        particles: &self.particles.colors,
                    })
                    .unwrap();
                let colors = colors_storage.into_inner();

                let sphere_count = self.particles.active_particles.len();

                //setting up the 3d rendering callback
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: std::sync::Arc::new(
                        eframe::egui_wgpu::CallbackFn::new()
                           //setting up for rendering data
                            .prepare(move |device, queue, encoder, paint_callback_resources| {
                                let renderer: &mut Renderer =
                                    paint_callback_resources.get_mut().unwrap();
                                renderer
                                    .update_resources(&camera, &particles, &colors, device, queue, encoder)
                            })
                            //rendering
                            .paint(move |_info, render_pass, paint_callback_resources| {
                                let renderer: &Renderer = paint_callback_resources.get().unwrap();
                                renderer.render(sphere_count as _, render_pass);
                            }),
                    ),
                });
                //updating the display continuously
                ctx.request_repaint();
            });
    }
}

//rendering handles the gpu drawing operations
struct Renderer {
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,//connect camera data to shaders
    particles_storage_buffer: wgpu::Buffer,// buffering for particle positions
    particles_storage_buffer_size: usize,// size tracking for efficient updates
    colors_storage_buffer: wgpu::Buffer,// buffering for particle colors
    colors_storage_buffer_size: usize,// size tracking for efficient updates
    particles_bind_group_layout: wgpu::BindGroupLayout,//connect particle data to shaders
    particles_bind_group: wgpu::BindGroup, //connection of particle data
    particles_render_pipeline: wgpu::RenderPipeline,//draw particles
    border_render_pipeline: wgpu::RenderPipeline,//draw world boundaries
}

impl Renderer {
    fn new(render_state: &eframe::egui_wgpu::RenderState) -> Self {
         // loading shader code for particles
        let particles_shader = render_state
            .device
            .create_shader_module(include_wgsl!("./particles.wgsl"));
        // loading shader code for world boundaries
        let border_shader = render_state
            .device
            .create_shader_module(include_wgsl!("./border.wgsl"));

         //camera data will be passing to shaders
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
        // creating a buffer for camera data
        let camera_uniform_buffer =
            render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Uniform Buffer"),
                    contents: &[0; <GpuCamera as ShaderSize>::SHADER_SIZE.get() as _],
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                });
        // connecting camera buffer to the shader
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

        //telling gpu how to access particle and color data
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

        //creating empty buffer to store particle positions on gpu for fast access
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

        //creating empty buffer to store particle colors on gpu for fast access
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

        //connecting our particle and color data to gpu memory
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

        //setting up how camera and particle data will flow through gpu
        let particles_pipeline_layout =
            render_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Particles Pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout, &particles_bind_group_layout],
                    push_constant_ranges: &[],
                });

        //setting up how particles will be drawn fast because gpu handles all particles in parallel
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

        //setting up how borders will use camera and particle data
        let border_pipeline_layout =
            render_state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Border Pipeline Layout"),
                    bind_group_layouts: &[&camera_bind_group_layout, &particles_bind_group_layout],
                    push_constant_ranges: &[],
                });

        //setting up how box borders will be drawn
        let border_render_pipeline = {
            let vertex_state = wgpu::VertexState {
                module: &border_shader,
                entry_point: "vs_main",
                buffers: &[],
            };
        
            let fragment_state = wgpu::FragmentState {
                module: &border_shader,
                entry_point: "fs_main",
                targets: &[Some(render_state.target_format.into())],
            };
        
            let primitive_state = wgpu::PrimitiveState {
                polygon_mode: wgpu::PolygonMode::Line,
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            };
        
            let depth_stencil_state = wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            };
        
            let multisample_state = wgpu::MultisampleState {
                ..Default::default()
            };
        
            render_state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Border Render Pipeline"),
                    layout: Some(&border_pipeline_layout),
                    vertex: vertex_state,
                    fragment: Some(fragment_state),
                    primitive: primitive_state,
                    depth_stencil: Some(depth_stencil_state),
                    multisample: multisample_state,
                    multiview: None,
                })
        };
        

        //collecting all the gpu memory and rendering pipelines
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

     //updating camera data on gpu
     fn update_resources(
        &mut self,
        camera_data: &[u8],
        particle_data: &[u8],
        color_data: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _cmd_encoder: &wgpu::CommandEncoder,
    ) -> Vec<wgpu::CommandBuffer> {
        //update camera
        queue.write_buffer(&self.camera_uniform_buffer, 0, camera_data);
        
        //track if we need to recreate the bind group
        let mut needs_bind_group_update = false;
        
        //handle particle buffer resizing with memory alignment to 4 bytes
        let particle_size_aligned = (particle_data.len() + 3) & !3;
        if self.particles_storage_buffer_size < particle_size_aligned {
            //apply growth factor of 1.2 to reduce future reallocations
            let target_size = ((particle_size_aligned as f32 * 1.2) as usize + 3) & !3;
            
            //create new buffer with increased capacity
            self.particles_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Data Buffer"),
                size: target_size as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            self.particles_storage_buffer_size = target_size;
            needs_bind_group_update = true;
        }
        
        //similar process for color buffer resizing
        let color_size_aligned = (color_data.len() + 3) & !3;
        if self.colors_storage_buffer_size < color_size_aligned {
            //calculate new buffer size with growth factor
            let target_size = ((color_size_aligned as f32 * 1.2) as usize + 3) & !3;
            
            //allocate new color buffer
            self.colors_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Color Buffer"),
                size: target_size as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            self.colors_storage_buffer_size = target_size;
            needs_bind_group_update = true;
        }
        
        //transfer the actual data to GPU memory
        queue.write_buffer(&self.particles_storage_buffer, 0, particle_data);
        queue.write_buffer(&self.colors_storage_buffer, 0, color_data);
        
        //regenerate bind group if buffer references changed
        if needs_bind_group_update {
            self.particles_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particle System Bind Group"),
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
        
        Vec::new()
    }
        

        

        //telling gpu which camera and particle data to use
        fn render<'a>(&'a self, particle_instances: u32, pass: &mut wgpu::RenderPass<'a>) {
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.particles_bind_group, &[]);
            
            if particle_instances > 0 {
                // First render the container borders
                pass.set_pipeline(&self.border_render_pipeline);
                pass.draw(0..24, 0..1);
                
                pass.set_pipeline(&self.particles_render_pipeline);
                pass.draw(0..4, 0..particle_instances);
            }
        }
}

fn main() {
    eframe::run_native(
        "3D Particle",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                present_mode: wgpu::PresentMode::AutoNoVsync,
                depth_format: Some(wgpu::TextureFormat::Depth32Float), //disable vsync for max speed            
                device_descriptor: wgpu::DeviceDescriptor {
                    features: wgpu::Features::POLYGON_MODE_LINE,
                    ..Default::default()
                },
                ..Default::default()
            },
            vsync: false,
            depth_buffer: 32,//turned off for faster rendering
            ..Default::default()
        },
        Box::new(|cc| Box::new(SimulationApp::new(cc))),
    )
    .unwrap();
}