#pragma once
// GPU-resident MLS-MPM: particles, the background grid and the diagnostics live on the GPU; the CPU
// orchestrates the four dispatches per substep and reads back a 64-byte diagnostics record.
#include "GpuApi.hpp"
#include "Math3D.hpp"
#include "Model.hpp"
#include "Shaders.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace material {
class Simulation {
  public:
    Settings settings;
    Diagnostics diagnostics{};
    float time = 0;  // simulated seconds since reset
    float gpuMs = 0; // most recently completed timing of one tick

    explicit Simulation(GpuApi &api) : gpu(api) {
        auto kernel = [&](std::string_view body, const char *label) {
            return gpu.program({{gl::COMPUTE_SHADER,
                                 {shaders::version, shaders::computeLayout, shaders::paramsBlock, shaders::pressHelper,
                                  shaders::common, body}}},
                               label);
        };
        programs.initialize = kernel(shaders::initialize, "Initialize");
        programs.clear = kernel(shaders::clear, "Clear grid");
        programs.p2g = kernel(shaders::p2g, "Particle to grid");
        programs.updateGrid = kernel(shaders::updateGrid, "Update grid");
        programs.g2p = kernel(shaders::g2p, "Grid to particle");
        programs.pick = kernel(shaders::pick, "GPU picking");
        programs.particles = gpu.program(
            {{gl::VERTEX_SHADER,
              {shaders::version, shaders::paramsBlock, shaders::viewBlock, shaders::common, shaders::particleVertex}},
             {gl::FRAGMENT_SHADER, {shaders::version, shaders::viewBlock, shaders::particleFragment}}},
            "Particle spheres");
        programs.scene = gpu.program(
            {{gl::VERTEX_SHADER,
              {shaders::version, shaders::paramsBlock, shaders::viewBlock, shaders::pressHelper, shaders::sceneVertex}},
             {gl::FRAGMENT_SHADER, {shaders::version, shaders::sceneFragment}}},
            "Scene");
        shapeLocation = gpu.GetUniformLocation(programs.scene, "shape");

        buffers.particles = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 0, maxParticles * sizeof(Particle), gl::DYNAMIC_COPY);
        buffers.accumulator = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 1, gridNodes * sizeof(F4), gl::DYNAMIC_COPY);
        buffers.velocities = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 2, gridNodes * sizeof(F4), gl::DYNAMIC_COPY);
        buffers.diagnostics = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 3, sizeof(Diagnostics), gl::DYNAMIC_COPY);
        buffers.pick = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 4, sizeof(std::uint32_t), gl::DYNAMIC_COPY);
        buffers.params = gpu.buffer(gl::UNIFORM_BUFFER, 0, sizeof(Params), gl::DYNAMIC_DRAW);
        buffers.view = gpu.buffer(gl::UNIFORM_BUFFER, 1, sizeof(ViewParams), gl::DYNAMIC_DRAW);
        gpu.GenVertexArrays(1, &emptyVao); // every mesh is generated from gl_VertexID / gl_InstanceID
        gpu.GenQueries(1, &query);
        reset();
    }
    ~Simulation() {
        gpu.DeleteQueries(1, &query);
        gpu.DeleteVertexArrays(1, &emptyVao);
        for (GLuint buffer : {buffers.particles, buffers.accumulator, buffers.velocities, buffers.diagnostics,
                              buffers.pick, buffers.params, buffers.view})
            gpu.DeleteBuffers(1, &buffer);
        for (GLuint program : {programs.initialize, programs.clear, programs.p2g, programs.updateGrid, programs.g2p,
                               programs.pick, programs.particles, programs.scene})
            gpu.DeleteProgram(program);
    }
    Simulation(const Simulation &) = delete;
    Simulation &operator=(const Simulation &) = delete;

    int count() const {
        return settings.count();
    }

    /// Re-samples the particles for the selected scene and clears the grid and diagnostics.
    void reset() {
        if (settings.side < 1 || settings.side > maxSide)
            throw std::runtime_error("Unsupported particle count.");
        if (timerPending) {
            std::uint64_t ignored{};
            gpu.GetQueryObjectui64v(query, gl::QUERY_RESULT, &ignored); // drain before reuse
            timerPending = false;
        }
        gpuMs = 0;
        time = 0;
        settings.dragging = false;
        uploadParams();
        dispatch(programs.initialize, count());
        dispatch(programs.clear, gridNodes);
        diagnostics = {};
        diagnostics.values[0] = static_cast<std::uint32_t>(count());
        gpu.MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        gpu.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffers.diagnostics);
        gpu.BufferSubData(gl::SHADER_STORAGE_BUFFER, 0, sizeof(diagnostics), &diagnostics);
    }

    /// Advances `substeps` substeps of 1/1200 s (one nominal tick by default), timed with a GPU query.
    void tick(int substeps = substepsPerTick) {
        pollTimer();
        const bool timing = !timerPending;
        if (timing)
            gpu.BeginQuery(gl::TIME_ELAPSED, query);
        for (int i = 0; i < substeps; ++i) {
            uploadParams(); // the press position and velocity depend on the substep time
            dispatch(programs.clear, gridNodes);
            dispatch(programs.p2g, count());
            dispatch(programs.updateGrid, gridNodes);
            dispatch(programs.g2p, count());
            time += stepSize;
        }
        if (timing) {
            gpu.EndQuery(gl::TIME_ELAPSED);
            timerPending = true;
        }
    }

    /// Reads back the 64-byte diagnostics record (a GPU sync point).
    void measure() {
        gpu.MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        gpu.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffers.diagnostics);
        gpu.GetBufferSubData(gl::SHADER_STORAGE_BUFFER, 0, sizeof(diagnostics), &diagnostics);
    }
    bool healthy() const {
        return diagnostics.invalid() == 0;
    }

    /// Distance along the cursor ray to the nearest particle, or infinity when nothing is hit.
    float pick(V3 origin, V3 direction) {
        std::uint32_t value = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
        gpu.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffers.pick);
        gpu.BufferSubData(gl::SHADER_STORAGE_BUFFER, 0, sizeof(value), &value);
        uploadParams(origin, direction);
        dispatch(programs.pick, count());
        gpu.MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        gpu.GetBufferSubData(gl::SHADER_STORAGE_BUFFER, 0, sizeof(value), &value);
        return std::bit_cast<float>(value);
    }

    /// Draws the floor, the scene collider and the instanced particle spheres.
    void draw(const Camera &camera, int width, int height) {
        uploadParams();
        uploadView(camera, static_cast<float>(width) / static_cast<float>(std::max(1, height)));
        gpu.BindVertexArray(emptyVao);
        glEnable(GL_DEPTH_TEST);
        gpu.UseProgram(programs.scene);
        gpu.Uniform1i(shapeLocation, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        if (settings.scene == Scene::sphere) {
            gpu.Uniform1i(shapeLocation, 1);
            glDrawArrays(GL_TRIANGLES, 0, 48 * 24 * 6);
        }
        if (settings.scene == Scene::press) {
            gpu.Uniform1i(shapeLocation, 2);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        gpu.UseProgram(programs.particles);
        gpu.DrawArraysInstanced(GL_TRIANGLES, 0, 6, count());
    }

  private:
    GpuApi &gpu;
    struct {
        GLuint initialize{}, clear{}, p2g{}, updateGrid{}, g2p{}, pick{}, particles{}, scene{};
    } programs;
    struct {
        GLuint particles{}, accumulator{}, velocities{}, diagnostics{}, pick{}, params{}, view{};
    } buffers;
    GLuint emptyVao{}, query{};
    GLint shapeLocation = -1;
    bool timerPending = false;

    void dispatch(GLuint program, int invocations) {
        gpu.UseProgram(program);
        gpu.DispatchCompute(static_cast<GLuint>(workgroups(invocations)), 1, 1);
        gpu.MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);
    }
    void uploadParams(V3 rayOrigin = {}, V3 rayDirection = {}) {
        const Params params{
            .count = count(),
            .resolution = gridSize,
            .side = settings.side,
            .materialType = std::to_underlying(settings.material),
            .scene = std::to_underlying(settings.scene),
            .tool = std::to_underlying(settings.tool),
            .dragging = settings.dragging ? 1 : 0,
            .colorMode = std::to_underlying(settings.color),
            .dt = stepSize,
            .young = settings.young,
            .gravity = settings.gravity,
            .friction = settings.friction,
            .brushRadius = settings.brushRadius,
            .time = time,
            .padding0 = 0,
            .padding1 = 0,
            .brush = settings.brush,
            .padding2 = 0,
            .rayOrigin = rayOrigin,
            .padding3 = 0,
            .rayDirection = rayDirection,
            .padding4 = 0,
        };
        gpu.BindBuffer(gl::UNIFORM_BUFFER, buffers.params);
        gpu.BufferSubData(gl::UNIFORM_BUFFER, 0, sizeof(params), &params);
    }
    void uploadView(const Camera &camera, float aspect) {
        const ViewParams view{.view = camera.view(), .projection = perspective(aspect)};
        gpu.BindBuffer(gl::UNIFORM_BUFFER, buffers.view);
        gpu.BufferSubData(gl::UNIFORM_BUFFER, 0, sizeof(view), &view);
    }
    /// Collects the previous tick's timing once the driver has it, without stalling.
    void pollTimer() {
        if (!timerPending)
            return;
        GLuint ready{};
        gpu.GetQueryObjectuiv(query, gl::QUERY_RESULT_AVAILABLE, &ready);
        if (!ready)
            return;
        std::uint64_t nanoseconds{};
        gpu.GetQueryObjectui64v(query, gl::QUERY_RESULT, &nanoseconds);
        gpuMs = static_cast<float>(nanoseconds) * 1e-6f;
        timerPending = false;
    }
};
} // namespace material
