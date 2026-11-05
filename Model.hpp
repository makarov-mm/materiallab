#pragma once
// Model constants and the GPU-visible records shared by the shaders, the simulation and the UI.
// The std140 structs mirror the uniform blocks in Shaders.hpp; the static_asserts pin their layout.
#include "Math3D.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace material {
using F4 = std::array<float, 4>;

constexpr int gridSize = 40;              // background grid nodes per axis
constexpr int gridNodes = gridSize * gridSize * gridSize;
constexpr int maxSide = 32;               // particles per axis of the initial cube (32^3 = 32,768)
constexpr int maxParticles = maxSide * maxSide * maxSide;
constexpr float stepSize = 1.f / 1200;    // seconds per substep
constexpr int substepsPerTick = 20;       // 20 substeps = one nominal 60 Hz tick
constexpr int tickRate = 60;
constexpr int workgroupSize = 128;        // layout(local_size_x) of every kernel

constexpr int workgroups(int invocations) {
    return (invocations + workgroupSize - 1) / workgroupSize;
}

enum class Material : int { snow, sand, clay };
enum class Scene : int { sphere, blocks, press };
enum class Tool : int { grab, cut, push };
enum class ColorMode : int { material, speed, volumeChange };

struct Settings {
    Material material = Material::snow;
    Scene scene = Scene::sphere;
    int side = 24;                       // particles per axis: 20, 24 or 32
    Tool tool = Tool::grab;
    ColorMode color = ColorMode::material;
    float young = 12000;                 // Young's modulus, simulation units
    float gravity = 2;
    float friction = .35f;               // collider contact friction
    float brushRadius = .09f;            // tool radius, domain units
    bool dragging = false;               // the tool is being held on the material
    V3 brush{};                          // tool position

    int count() const {
        return side * side * side;
    }
};

/// One material point, 128 bytes in std430: position (w: active), velocity (w: plastic volume
/// history), the deformation gradient columns and the APIC affine velocity columns.
struct alignas(16) Particle {
    F4 x, v, f0, f1, f2, c0, c1, c2;
};

/// 64-byte diagnostics buffer (SSBO binding 3), accumulated by the kernels and read twice per second.
struct Diagnostics {
    std::array<std::uint32_t, 16> values{};

    std::uint32_t active() const {
        return values[0];
    }
    std::uint32_t invalid() const {
        return values[1];
    }
    std::uint32_t speedClamps() const {
        return values[2];
    }
    std::uint32_t boundaryCorrections() const {
        return values[3];
    }
    float maxSpeed() const {
        return std::bit_cast<float>(values[4]);
    }
};

/// std140 mirror of shaders::paramsBlock (uniform binding 0).
struct alignas(16) Params {
    std::int32_t count, resolution, side, materialType;
    std::int32_t scene, tool, dragging, colorMode;
    float dt, young, gravity, friction;
    float brushRadius, time, padding0, padding1;
    V3 brush;
    float padding2;
    V3 rayOrigin;
    float padding3;
    V3 rayDirection;
    float padding4;
};

/// std140 mirror of shaders::viewBlock (uniform binding 1).
struct alignas(16) ViewParams {
    M4 view, projection;
};

static_assert(sizeof(Particle) == 128 && sizeof(Diagnostics) == 64);
static_assert(sizeof(Params) == 112 && offsetof(Params, brush) == 64 && offsetof(Params, rayOrigin) == 80 &&
              offsetof(Params, rayDirection) == 96);
static_assert(sizeof(ViewParams) == 128);
} // namespace material
