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
    let distance = length(in.uv * 2.0 - 1.0);
    if distance > 1.0 {
        discard;
    }
    
    let particle_id = particles.particles[in.particle_index].id;
    let color = colors.colors[particle_id];
    
    return vec4(color, 1.0);
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let vertex_index = in.vertex_index;
    let particle_index = in.particle_index;
    
    let u = f32((vertex_index >> 0u) & 1u);
    let v = f32((vertex_index >> 1u) & 1u);
    let uv = vec2(u, v);
    
    let particle_pos = particles.particles[particle_index].position;
    let view_space_pos = camera.view_matrix * vec4(particle_pos, 1.0);
    
    let offset = vec4((uv - 0.5) * 0.1, 0.0, 0.0);
    let final_view_pos = view_space_pos + offset;
    
    let clip_pos = camera.projection_matrix * final_view_pos;
    let world_pos = clip_pos.xyz / clip_pos.w;
    
    var out: VertexOut;
    out.particle_index = particle_index;
    out.uv = uv;
    out.position = clip_pos;
    out.world_position = world_pos;
    
    return out;
}