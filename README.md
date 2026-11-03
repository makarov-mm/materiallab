# Material Lab — 3D GPU Edition

![](screenshot.png)

**Interface update:** This project uses the shared modern WinAPI interface: flat themed controls, scrollable settings and keyboard navigation. See [Modern UI](../../MODERN_UI.md). Simulation shortcuts apply when the 3D canvas has focus; focused controls use their native keys.

A native Windows laboratory for three-dimensional deformable materials. The complete simulation runs in OpenGL 4.3 compute shaders. C++23 supplies the WinAPI interface, camera, WGL context, dispatch scheduling and diagnostics. No third-party library, package manager, model, texture or external shader file is required.

## Build and run

Open **MaterialLab3D.sln** at the repository root with Visual Studio 2026, select **Release | x64**, build and run 01_MaterialLab. Install the Desktop development with C++ workload, MSVC v145 and a Windows SDK. The full PhysicsLab.sln also builds the 3D application as its first project.

Alternatively run BuildMaterial3D.ps1. The executable is bin/x64/Release/01_MaterialLab.exe. The build uses C++23 preview mode on MSVC 14.51, a static C++ runtime, and Windows system import libraries only. Debug information uses /Z7 to avoid a compiler PDB server. An OpenGL 4.3-capable driver and a desktop large enough for the minimum 1040 × 920 window are required.

## Materials and scenes

- **Snow:** fixed-corotated elasticity, singular-value plastic projection and bounded exponential hardening.
- **Sand:** logarithmic elastic strain, Hencky stress and a noncohesive Drucker–Prager projection with a fixed 30-degree internal friction angle.
- **Clay:** fixed-corotated stress with bounded logarithmic volumetric strain and a deviatoric plastic-yield envelope.

Drop material onto a sphere, collide two blocks, or compress a block with a moving horizontal press. The domain includes frictional floor and side contacts. The press is kinematic and does not respond to reaction forces.

Choose 8,000, 13,824 or 32,768 particles. Each resolution represents the same initial material volume. Changing material, scene or particle count resets the experiment. Physical sliders preserve the current state.

## Controls

| Control | Action |
|---|---|
| Right mouse drag | Orbit camera |
| Mouse wheel | Zoom |
| Left mouse drag | Apply selected tool to visible material |
| Grab | Spring-like attraction and local damping |
| Cut | Delete particles inside the tool sphere |
| Push | Radial repulsive force |
| Space / Pause | Pause or resume |
| R / Reset | Reinitialize experiment |
| N / Step | Pause and advance one 1/1200-second substep |
| Reset camera | Restore original orbit |
| Escape | Close |

A GPU ray query selects the nearest particle on mouse-down and establishes a camera-facing interaction plane. Dragging moves the tool on that plane, preserving the selected depth. This hit test reads back one 32-bit depth value. Clicking empty space does not activate a tool. Tools act only while physics advances.

Slider ranges: Young's modulus 2,000–29,000; gravity 0–5; contact friction 0–1; tool radius 0.03–0.18. Density is 1,000. Initial volume is 0.36³ and particle mass equals density times particle volume. These are simulation units, not calibrated material measurements. Collider friction and sand's fixed internal friction angle are separate parameters.

## GPU pipeline

Particles and the 40 × 40 × 40 background grid live in shader storage buffer objects. Each particle occupies 128 bytes in std430: position/active flag, velocity/plastic-volume history, three padded deformation-gradient columns and three padded APIC-affine columns.

Each substep consists of four compute dispatches:

1. Clear grid momentum, mass and velocity.
2. **P2G:** evaluate 3D elastic stress and transfer mass, momentum and APIC affine motion to a 27-node quadratic B-spline stencil.
3. **Grid update:** normalize momentum; apply gravity, tool forces and frictional collision response.
4. **G2P:** interpolate velocity, reconstruct the affine field, advect particles, evolve the 3 × 3 deformation gradient and apply plastic projection.

The GPU computes the 3D SVD using five Jacobi sweeps on FᵀF. Every dispatch ends with GL_SHADER_STORAGE_BARRIER_BIT. Readbacks additionally use GL_BUFFER_UPDATE_BARRIER_BIT.

OpenGL 4.3 does not require native float atomic addition. P2G therefore implements atomic float sums with integer atomicCompSwap on SSBO words, including an atomic initial read. No vendor extension is required. This avoids fixed-point quantization and accumulation range limits. Concurrent summation order can vary, so results are not bitwise deterministic across runs or GPUs.

The fixed time step is 1/1200 second. Twenty substeps form a nominal 60 Hz simulation tick. Slow hardware slows simulated time rather than increasing the integration step. All physics runs on the GPU, including Debug builds; there is no CPU solver fallback.

## Rendering and diagnostics

Instanced sphere impostors read particle positions directly from the SSBO. The fragment shader reconstructs normals and writes corrected sphere depth. Perspective, lighting, material/velocity/deformation coloring, a procedural floor and a lit collider make the 3D structure visible. Particle positions are not copied back to the CPU for rendering.

The status panel reports active particles, GPU physics time from timer queries, render FPS, simulated time, maximum speed, invalid states, speed caps and boundary corrections. A 64-byte diagnostics buffer is read twice per second. Error/correction counts accumulate until reset. Invalid state pauses interactive simulation at the next diagnostic check. Speed is capped at 8; final position clamping is an additional reported safeguard.

## Verification

Run **TestMaterial3D.ps1** to test the actual compute shaders. The suite covers all nine material/scene combinations, both particle-count extremes at maximum stiffness/gravity/friction, mass transfer, finite state, gravity response, zero-gravity symmetry, cutting, picking and GL errors. Full-buffer readback is used only in tests. Results are in Validation3D/Material3D-tests.txt and Validation3D/REPORT.md.

The executable also accepts --smoke (120 simulation ticks, a BMP and a report) and --gpu-test (extended suite). Both write to their working directory and return nonzero on failure. They require a working OpenGL desktop context.

The collection's old Simulation.cpp remains a CPU 2D reference for LabTests. It is not linked into this 3D application and is excluded from the standalone 3D archive.

## Numerical scope

This is a fully 3D GPU simulator, not a physically calibrated engineering package. Clay's yield law is an educational approximation. Cutting removes mass rather than modeling fracture energy. Initial sampling uses a regular lattice. There is no watertight surface reconstruction, self-shadowing, adaptive CFL step, sparse grid, export, or mixed-material scene. Subgrid fracture and contact depend on grid resolution. Sphere impostors approximate silhouettes and are intended for particles small relative to camera distance.

## Source map

- Material3D.cpp: dispatch, WinAPI UI, interaction, rendering and GPU tests.
- Shaders.hpp: embedded GLSL 4.30 compute, vertex and fragment shaders.
- GpuApi.hpp: WGL/OpenGL loading and resource ownership.
- Math3D.hpp: camera and projection math.

References (not dependencies): [MPM snow](https://www.disneyanimation.com/publications/a-material-point-method-for-snow-simulation/), [MLS-MPM course](https://yuanming.taichi.graphics/publication/2019-mpm-tutorial/mpm-course-2019.pdf), [OpenGL compute](https://wikis.khronos.org/opengl/Compute_Shader), [SSBO atomics](https://wikis.khronos.org/opengl/Shader_Storage_Buffer_Object).

