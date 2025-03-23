struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    id: u32,
};

struct Particles {
    world_size: f32,
    length: u32,
    particles: array<Particle>,
};

struct Colors {
    length: u32,
    colors: array<vec3<f32>>,
};

struct Camera {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
};

var<private> quad_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-0.5, 0.5),  // Top-left
    vec2<f32>(-0.5, -0.5), // Bottom-left
    vec2<f32>(0.5, 0.5),   // Top-right
    vec2<f32>(0.5, -0.5)   // Bottom-right
);

var<private> quad_uvs: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0)
);

struct VertexIn {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) particle_index: u32,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) particle_index: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> particles: Particles;
@group(1) @binding(1) var<storage, read> colors: Colors;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let radius_check = length(in.uv * 2.0 - 1.0);
    if radius_check > 1.0 {
        discard;
    }
    
    let particle = particles.particles[in.particle_index];
    let color = colors.colors[particle.id];
    
    return vec4(color, 1.0);
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let vertex_pos = quad_vertices[in.vertex_index % 4u];
    let particle_idx = in.particle_index;
    
    let uv = quad_uvs[in.vertex_index % 4u];
    
    let particle_position = particles.particles[particle_idx].position;
    
    let view_pos = camera.view_matrix * vec4(particle_position, 1.0);
    
    let scaled_vertex = vertex_pos * 0.1;
    let billboard_pos = view_pos + vec4(scaled_vertex, 0.0, 0.0);
    
    let clip_pos = camera.projection_matrix * billboard_pos;
    
    var out: VertexOut;
    out.position = clip_pos;
    out.world_position = clip_pos.xyz / clip_pos.w;
    out.uv = uv;
    out.particle_index = particle_idx;
    
    return out;
}