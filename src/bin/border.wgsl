struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    id: u32,
};

//holds all particles in one big structure
struct Particles {
    world_size: f32, //size of the world the particles live in
    length: u32, //how many particles I have
    particles: array<Particle>, //actual particle data
};

//stores colors for particles
struct Colors {
    length: u32,
    colors: array<vec3<f32>>, //rgb color values
};

//camera info for transforming particles
struct Camera {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>, //transforms camera space to clip space
};

//defines corners of particle quads
var<private> quad_vertices: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-0.5, 0.5),  // Top-left
    vec2<f32>(-0.5, -0.5), // Bottom-left
    vec2<f32>(0.5, 0.5),   // Top-right
    vec2<f32>(0.5, -0.5)   // Bottom-right
);

//coordinates for quads
var<private> quad_uvs: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),  //top-left uv
    vec2<f32>(0.0, 1.0), //bottom-left uv
    vec2<f32>(1.0, 0.0), //top-right uv
    vec2<f32>(1.0, 1.0) //bottom-right uv
);

//inputs to vertex shader
struct VertexIn {
    @builtin(vertex_index) vertex_index: u32, //which vertex is processing
    @builtin(instance_index) particle_index: u32, //which particle this vertex belongs to
};

//outputs from vertex shader to fragment shader
struct VertexOut {
    @builtin(position) position: vec4<f32>, //final clip space position
    @location(0) world_position: vec3<f32>, //position in world
    @location(1) uv: vec2<f32>, //texture coordinates
    @location(2) particle_index: u32, //which particle this is
};

//shader uniforms and storage buffers
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<storage, read> particles: Particles;
@group(1) @binding(1) var<storage, read> colors: Colors;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    //check if pixel is inside circle
    let radius_check = length(in.uv * 2.0 - 1.0);
    if radius_check > 1.0 {
        discard;  //don't draw pixels outside circle
    }
        //getting this particle's info
    let particle = particles.particles[in.particle_index];
        //getting color based on particle id
    let color = colors.colors[particle.id];
    
    return vec4(color, 1.0);     //return final color with full opacity
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
        //getting which corner of quad drawing
    let vertex_pos = quad_vertices[in.vertex_index % 4u];
        //getting which particle rendering
    let particle_idx = in.particle_index;
    
    //getting texture coordinate for this vertex
    let uv = quad_uvs[in.vertex_index % 4u];
    
    //getting position of this particle
    let particle_position = particles.particles[particle_idx].position;
    
    let view_pos = camera.view_matrix * vec4(particle_position, 1.0);
    
    let scaled_vertex = vertex_pos * 0.1;
    let billboard_pos = view_pos + vec4(scaled_vertex, 0.0, 0.0);
    
    //transform to clip space
    let clip_pos = camera.projection_matrix * billboard_pos;
    
    //setup output
    var out: VertexOut;
    out.position = clip_pos;
    out.world_position = clip_pos.xyz / clip_pos.w;
    out.uv = uv;
    out.particle_index = particle_idx;
    
    return out;
}
//this shader has good runtime because:
    //it process many particles in parallel and avoids costly matrix math per vertex
    //it discard earlier in fragment shader saves pixel processing and use direct array access for particle data which is fast