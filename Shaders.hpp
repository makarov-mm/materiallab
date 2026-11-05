#pragma once
// GLSL 4.30 sources. Programs are assembled from chunks (see Simulation.hpp) so that the uniform
// blocks are declared once and stay in sync with the structs in Model.hpp.
#include <string_view>

namespace material::shaders {
inline constexpr std::string_view version = "#version 430 core\n";
inline constexpr std::string_view computeLayout = "layout(local_size_x = 128) in;\n";

// Simulation, tool, display and picking parameters; mirrors material::Params (std140, binding 0).
inline constexpr std::string_view paramsBlock = R"GLSL(
layout(std140, binding = 0) uniform Params {
    int count, resolution, side, materialType;
    int scene, tool, dragging, colorMode;
    float dt, young, gravity, friction;
    float brushRadius, time;
    vec3 brush;
    vec3 rayOrigin;
    vec3 rayDirection;
};
)GLSL";

// Camera; mirrors material::ViewParams (std140, binding 1).
inline constexpr std::string_view viewBlock = R"GLSL(
layout(std140, binding = 1) uniform View {
    mat4 view, projection;
};
)GLSL";

// Height of the kinematic press plate at the current time (scene 2); used by physics and rendering.
inline constexpr std::string_view pressHelper = R"GLSL(
float pressHeight() { return .84 - .60 * (.5 - .5 * cos(time * 1.4)); }
)GLSL";

// Storage, the MLS-MPM helpers and the 3x3 SVD shared by the kernels and the particle renderer.
inline constexpr std::string_view common = R"GLSL(
struct Particle { vec4 x; vec4 v; vec4 f0; vec4 f1; vec4 f2; vec4 c0; vec4 c1; vec4 c2; };   // see material::Particle
layout(std430, binding = 0) buffer Particles { Particle particles[]; };
layout(std430, binding = 1) buffer Accumulator { uvec4 grid[]; };     // momentum xyz, mass; float bits for atomics
layout(std430, binding = 2) buffer Velocities { vec4 velocity[]; };  // node velocity, mass
layout(std430, binding = 3) buffer Diagnostics { uint stats[]; };    // see material::Diagnostics
layout(std430, binding = 4) buffer Picking { uint picked; };         // float bits of the nearest hit distance

const float density = 1000.0;
float spacing() { return .36 / float(side); }
float volume() { float s = spacing(); return s * s * s; }
uint index3(ivec3 p) { return uint((p.z * resolution + p.y) * resolution + p.x); }
mat3 deformation(Particle p) { return mat3(p.f0.xyz, p.f1.xyz, p.f2.xyz); }
mat3 affine(Particle p) { return mat3(p.c0.xyz, p.c1.xyz, p.c2.xyz); }
void storeF(inout Particle p, mat3 f) { p.f0 = vec4(f[0], 0); p.f1 = vec4(f[1], 0); p.f2 = vec4(f[2], 0); }
void storeC(inout Particle p, mat3 c) { p.c0 = vec4(c[0], 0); p.c1 = vec4(c[1], 0); p.c2 = vec4(c[2], 0); }
mat3 diagonal(vec3 d) { return mat3(d.x, 0, 0, 0, d.y, 0, 0, 0, d.z); }
bool invalid(vec3 v) { return any(isnan(v)) || any(isinf(v)); }

// One Jacobi rotation that zeroes a[q][p] of the symmetric matrix a, accumulated into v.
void rotateJacobi(inout mat3 a, inout mat3 v, int p, int q) {
    float off = a[q][p];
    if (abs(off) < 1e-8) return;
    float tau = (a[q][q] - a[p][p]) / (2 * off);
    float t = (tau >= 0 ? 1.0 : -1.0) / (abs(tau) + sqrt(1 + tau * tau));
    float c = inversesqrt(1 + t * t), s = t * c;
    mat3 j = mat3(1);
    j[p][p] = c; j[q][q] = c; j[p][q] = -s; j[q][p] = s;
    a = transpose(j) * a * j;
    v = v * j;
}
// SVD of f from five Jacobi sweeps on f^T f: f = u * diag(sigma) * v^T.
void svd(mat3 f, out mat3 u, out vec3 sigma, out mat3 v) {
    mat3 a = transpose(f) * f;
    v = mat3(1);
    for (int i = 0; i < 5; ++i) { rotateJacobi(a, v, 0, 1); rotateJacobi(a, v, 0, 2); rotateJacobi(a, v, 1, 2); }
    sigma = sqrt(max(vec3(a[0][0], a[1][1], a[2][2]), vec3(1e-10)));
    u = f * v * diagonal(1.0 / sigma);
}
// Quadratic B-spline weights for the three nodes around a fractional position.
void weights(vec3 f, out vec3 w[3]) {
    w[0] = .5 * (1.5 - f) * (1.5 - f);
    w[1] = .75 - (f - 1) * (f - 1);
    w[2] = .5 * (f - .5) * (f - .5);
}
// Frictional contact against a wall moving with wallVelocity (Coulomb, no separation force).
vec3 contact(vec3 v, vec3 normal, vec3 wallVelocity) {
    vec3 relative = v - wallVelocity;
    float vn = dot(relative, normal);
    if (vn >= 0) return v;
    vec3 tangent = relative - vn * normal;
    float speed = length(tangent);
    return wallVelocity + tangent * max(0.0, 1.0 - friction * (-vn) / max(speed, 1e-8));
}
)GLSL";

// Regular lattice inside a 0.36 cube, placed per scene; identity deformation, zero affine field.
inline constexpr std::string_view initialize = R"GLSL(
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(count)) return;
    int n = side;
    ivec3 ijk = ivec3(int(id) % n, (int(id) / n) % n, int(id) / (n * n));
    vec3 position = vec3(.32, .53, .32) + (vec3(ijk) + .5) * spacing();
    vec3 v = vec3(0);
    if (scene == 1) {   // two half blocks approaching each other
        bool left = ijk.x < n / 2;
        position.x = left ? .18 + (float(ijk.x) + .5) * spacing() : .64 + (float(ijk.x - n / 2) + .5) * spacing();
        position.y -= .16;
        v.x = left ? .7 : -.7;
    }
    if (scene == 2) position.y -= .38;   // on the floor, under the press
    Particle p;
    p.x = vec4(position, 1);
    p.v = vec4(v, 1);
    storeF(p, mat3(1));
    storeC(p, mat3(0));
    particles[id] = p;
}
)GLSL";

inline constexpr std::string_view clear = R"GLSL(
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(resolution * resolution * resolution)) return;
    grid[id] = uvec4(0);
    velocity[id] = vec4(0);
    if (id == 0u || id == 4u || id == 5u) stats[id] = 0u;   // per-substep counters and maxima
}
)GLSL";

// Particle to grid: elastic stress and APIC transfer onto the 27-node quadratic stencil.
inline constexpr std::string_view p2g = R"GLSL(
// OpenGL 4.3 has no float atomics: integer compare-and-swap implements the atomic float sum.
// The initial value is read atomically too, so ordinary reads never mix with concurrent updates.
void atomicFloatAdd(uint node, int component, float value) {
    uint old = atomicCompSwap(grid[node][component], 0u, 0u);
    for (;;) {
        uint expected = old;
        uint next = floatBitsToUint(uintBitsToFloat(expected) + value);
        old = atomicCompSwap(grid[node][component], expected, next);
        if (old == expected) break;
    }
}
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(count)) return;
    Particle p = particles[id];
    if (p.x.w == 0) return;   // cut away
    mat3 f = deformation(p), u, v;
    vec3 sigma;
    svd(f, u, sigma, v);
    float hard = materialType == 0 ? exp(clamp(7 * (1 - p.v.w), -1.0, 1.0)) : 1;   // snow hardening
    float e = young * hard, mu = e / (2 * 1.2), lambda = e * .2 / (1.2 * .6);
    float j = determinant(f);
    mat3 stress;
    if (materialType == 1) {   // sand: Hencky stress from the logarithmic strain
        vec3 eps = log(max(sigma, vec3(1e-5)));
        stress = u * diagonal(2 * mu * eps + lambda * (eps.x + eps.y + eps.z)) * transpose(u);
    } else {                   // snow and clay: fixed corotated
        mat3 rotation = u * transpose(v);
        stress = 2 * mu * (f - rotation) * transpose(f) + mat3(lambda * j * (j - 1));
    }
    float invDx = float(resolution), mass = density * volume();
    mat3 apic = mass * affine(p) - dt * volume() * 4 * invDx * invDx * stress;
    vec3 xp = p.x.xyz * invDx;
    ivec3 base = ivec3(floor(xp - .5));
    vec3 fpos = xp - vec3(base), w[3];
    weights(fpos, w);
    for (int z = 0; z < 3; ++z)
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x) {
                ivec3 cell = base + ivec3(x, y, z);
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, ivec3(resolution)))) continue;
                float weight = w[x].x * w[y].y * w[z].z;
                vec3 momentum = weight * (mass * p.v.xyz + apic * ((vec3(x, y, z) - fpos) / invDx));
                uint node = index3(cell);
                atomicFloatAdd(node, 0, momentum.x);
                atomicFloatAdd(node, 1, momentum.y);
                atomicFloatAdd(node, 2, momentum.z);
                atomicFloatAdd(node, 3, weight * mass);
            }
}
)GLSL";

// Grid update: velocity from momentum, gravity, tool forces and frictional collisions.
inline constexpr std::string_view updateGrid = R"GLSL(
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(resolution * resolution * resolution)) return;
    vec4 g = uintBitsToFloat(grid[id]);
    if (g.w < 1e-12) return;
    ivec3 cell = ivec3(int(id) % resolution, (int(id) / resolution) % resolution, int(id) / (resolution * resolution));
    vec3 x = vec3(cell) / float(resolution), v = g.xyz / g.w;
    v.y -= gravity * dt;
    if (dragging != 0 && tool != 1) {   // grab: spring and damping; push: radial repulsion
        vec3 d = brush - x;
        float r = length(d);
        if (r < brushRadius) {
            float influence = 1 - r / brushRadius;
            vec3 force = tool == 0 ? 80 * d - 5 * v : -20 * d / max(r, .01);
            v += dt * force * influence;
        }
    }
    if (scene == 0) {   // the sphere collider
        vec3 d = x - vec3(.5, .28, .5);
        float r = length(d);
        if (r < .145 && r > 1e-6) v = contact(v, d / r, vec3(0));
    }
    if (scene == 2 && x.y > pressHeight()) v = contact(v, vec3(0, -1, 0), vec3(0, -.42 * sin(time * 1.4), 0));
    for (int axis = 0; axis < 3; ++axis) {   // domain walls
        vec3 normal = vec3(0);
        normal[axis] = 1;
        if (cell[axis] < 3) v = contact(v, normal, vec3(0));
        if (cell[axis] > resolution - 4) v = contact(v, -normal, vec3(0));
    }
    velocity[id] = vec4(v, g.w);
}
)GLSL";

// Grid to particle: velocity and affine reconstruction, advection, deformation update and plasticity.
inline constexpr std::string_view g2p = R"GLSL(
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(count)) return;
    Particle p = particles[id];
    if (p.x.w == 0) return;
    if (dragging != 0 && tool == 1 && distance(p.x.xyz, brush) < brushRadius) {   // cut
        p.x.w = 0;
        particles[id] = p;
        return;
    }
    float invDx = float(resolution);
    vec3 xp = p.x.xyz * invDx;
    ivec3 base = ivec3(floor(xp - .5));
    vec3 fpos = xp - vec3(base), w[3];
    weights(fpos, w);
    vec3 speed = vec3(0);
    mat3 c = mat3(0);
    for (int z = 0; z < 3; ++z)
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x) {
                ivec3 cell = base + ivec3(x, y, z);
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, ivec3(resolution)))) continue;
                float weight = w[x].x * w[y].y * w[z].z;
                vec3 v = velocity[index3(cell)].xyz;
                speed += weight * v;
                c += 4 * invDx * weight * outerProduct(v, vec3(x, y, z) - fpos);
            }
    mat3 trial = (mat3(1) + dt * c) * deformation(p), u, v;
    vec3 sigma;
    svd(trial, u, sigma, v);
    vec3 projected = sigma;
    if (materialType == 0) {   // snow: clamp the stretches, track the plastic volume for hardening
        projected = clamp(sigma, vec3(.975), vec3(1.0075));
        p.v.w = clamp(p.v.w * (sigma.x * sigma.y * sigma.z) / (projected.x * projected.y * projected.z), .6, 1.4);
    }
    if (materialType == 1) {   // sand: Drucker-Prager with a ~30 degree friction angle
        vec3 eps = log(max(sigma, vec3(1e-5)));
        float tr = eps.x + eps.y + eps.z;
        vec3 dev = eps - tr / 3;
        float normDev = length(dev);
        float alpha = .3265986;
        if (tr >= 0) eps = vec3(0);
        else if (normDev > 1e-8) {
            float delta = normDev + 2.0 * tr * alpha;
            if (delta > 0) eps -= delta * dev / normDev;
        }
        projected = exp(eps);
    }
    if (materialType == 2) {   // clay: bounded volumetric strain and a deviatoric yield envelope
        vec3 eps = log(max(sigma, vec3(1e-5)));
        float mean = (eps.x + eps.y + eps.z) / 3;
        vec3 dev = eps - mean;
        float lengthDev = length(dev);
        eps = vec3(clamp(mean, -.12, .12)) + dev * min(1.0, .065 / max(lengthDev, 1e-8));
        projected = exp(eps);
    }
    mat3 f = u * diagonal(projected) * transpose(v);
    bool bad = invalid(speed) || invalid(f[0]) || invalid(f[1]) || invalid(f[2]) || determinant(trial) <= 0;
    if (bad) { atomicAdd(stats[1], 1u); return; }
    if (length(speed) > 8) { speed = normalize(speed) * 8; atomicAdd(stats[2], 1u); }
    p.v.xyz = speed;
    p.x.xyz += dt * speed;
    float lo = 2.1 / invDx, hi = 1 - lo;
    for (int a = 0; a < 3; ++a)
        if (p.x[a] < lo || p.x[a] > hi) {
            p.x[a] = clamp(p.x[a], lo, hi);
            p.v[a] = 0;
            atomicAdd(stats[3], 1u);
        }
    storeF(p, f);
    storeC(p, c);
    particles[id] = p;
    atomicAdd(stats[0], 1u);
    atomicMax(stats[4], floatBitsToUint(length(speed)));
    atomicMax(stats[5], floatBitsToUint(abs(determinant(f) - 1)));
}
)GLSL";

// Nearest particle along the cursor ray, as the minimum hit distance (float bits in a uint).
inline constexpr std::string_view pick = R"GLSL(
void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(count)) return;
    Particle p = particles[id];
    if (p.x.w == 0) return;
    vec3 d = p.x.xyz - rayOrigin;
    float t = dot(d, rayDirection);
    float r = length(d - t * rayDirection);
    if (t > 0 && r < spacing() * .8) atomicMin(picked, floatBitsToUint(t));
}
)GLSL";

// Instanced sphere impostors: one camera-facing quad per particle.
inline constexpr std::string_view particleVertex = R"GLSL(
out vec2 uv;
out vec3 centerView;
out vec3 tint;
flat out float radius;
void main() {
    Particle p = particles[gl_InstanceID];
    const vec2 corners[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    uv = corners[gl_VertexID];
    radius = spacing() * .64;
    centerView = (view * vec4(p.x.xyz, 1)).xyz;
    tint = materialType == 0 ? vec3(.78, .88, .98) : materialType == 1 ? vec3(.82, .54, .22) : vec3(.64, .22, .14);
    if (colorMode == 1) tint = mix(vec3(.10, .45, .85), vec3(1, .28, .08), clamp(length(p.v.xyz) / 2, 0, 1));
    if (colorMode == 2) tint = mix(vec3(.12, .75, .6), vec3(1, .18, .1), clamp(abs(determinant(deformation(p)) - 1) * 8, 0, 1));
    if (p.x.w == 0) { gl_Position = vec4(3, 3, 3, 1); return; }   // cut away: clipped
    gl_Position = projection * vec4(centerView + vec3(uv * radius, 0), 1);
}
)GLSL";

// Sphere shading with the corrected depth of the impostor surface.
inline constexpr std::string_view particleFragment = R"GLSL(
in vec2 uv;
in vec3 centerView;
in vec3 tint;
flat in float radius;
out vec4 color;
void main() {
    float r2 = dot(uv, uv);
    if (r2 > 1) discard;
    vec3 normal = vec3(uv, sqrt(1 - r2));
    vec3 light = normalize(mat3(view) * vec3(-.4, .8, .55));
    float diffuse = max(dot(normal, light), 0);
    float spec = pow(max(dot(normal, normalize(light + vec3(0, 0, 1))), 0), 40);
    vec3 rgb = tint * (.26 + .74 * diffuse) + vec3(.16 * spec);
    color = vec4(pow(rgb, vec3(1 / 2.2)), 1);
    vec4 clip = projection * vec4(centerView + radius * normal, 1);
    gl_FragDepth = clip.z / clip.w * .5 + .5;
}
)GLSL";

// Floor (shape 0), the sphere collider (1) and the press plate (2), all from gl_VertexID.
inline constexpr std::string_view sceneVertex = R"GLSL(
uniform int shape;
out vec3 world;
out vec3 normal;
const vec2 corner[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));
void main() {
    vec2 uv = corner[gl_VertexID % 6];
    if (shape == 0) {
        world = vec3(mix(-.15, 1.15, uv.x), .0525, mix(-.15, 1.15, uv.y));
        normal = vec3(0, 1, 0);
    } else if (shape == 1) {   // 48 x 24 sphere tiles
        int tile = gl_VertexID / 6;
        float a = (float(tile % 48) + uv.x) / 48 * 6.2831853, b = (float(tile / 48) + uv.y) / 24 * 3.14159265;
        normal = vec3(sin(b) * cos(a), cos(b), sin(b) * sin(a));
        world = vec3(.5, .28, .5) + normal * .145;
    } else {
        world = vec3(mix(.06, .94, uv.x), pressHeight(), mix(.06, .94, uv.y));
        normal = vec3(0, -1, 0);
    }
    gl_Position = projection * view * vec4(world, 1);
}
)GLSL";

inline constexpr std::string_view sceneFragment = R"GLSL(
uniform int shape;
in vec3 world;
in vec3 normal;
out vec4 color;
void main() {
    vec3 base = shape == 1 ? vec3(.12, .20, .28) : shape == 2 ? vec3(.3, .34, .4) : vec3(.055, .075, .10);
    float lighting = .4 + .6 * max(dot(normalize(normal), normalize(vec3(-.4, .8, .55))), 0);
    if (shape == 0) {   // floor grid lines
        vec2 grid = abs(fract(world.xz * 20 - .5) - .5) / max(fwidth(world.xz * 20), vec2(.001));
        float line = 1 - min(min(grid.x, grid.y), 1);
        base += line * .045;
    }
    color = vec4(pow(base * lighting, vec3(1 / 2.2)), 1);
}
)GLSL";
} // namespace material::shaders
