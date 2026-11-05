#pragma once
// Native window with an OpenGL viewport child, settings panel, camera and tool interaction, status line.
#include "GpuApi.hpp"
#include "Math3D.hpp"
#include "Model.hpp"
#include "Simulation.hpp"
#include <array>
#include <memory>

namespace material {
constexpr int panelWidth = 320;   // settings column
constexpr int statusHeight = 64;  // status strip under the viewport
constexpr int sliderCount = 4, comboCount = 5;
constexpr int initialWidth = 1360, initialHeight = 1000, minimumWidth = 1080, minimumHeight = 760;
constexpr const wchar_t *appTitle = L"Material Lab", *appSubtitle = L"Snow, sand and clay on the GPU";

/// Control identifiers. Combos and sliders are consecutive so the index is id - first.
enum class Id : int {
    material = 10, experiment, particles, tool, color, //
    slider = 20,                                       // slider + 0..3
    pause = 30, reset, step, resetCamera
};

/// One settings slider: linear map from the 0..100 trackbar position to [low, high].
struct SliderSpec {
    const wchar_t *label;
    float low, high;
    int initial;             // trackbar position 0..100
    const wchar_t *format;   // std::format specification for the value
    float Settings::*target; // the Settings member driven by the slider
};
extern const std::array<SliderSpec, sliderCount> sliderSpecs;

struct App {
    GpuApi gpu;
    std::unique_ptr<Simulation> sim;
    Camera camera;
    HWND window{}, viewport{}, status{}, pauseButton{};
    std::array<HWND, comboCount> combos{};
    std::array<HWND, sliderCount> sliders{}, labels{};
    int width = initialWidth, height = initialHeight; // root client size
    bool running = true, paused = false, single = false, orbiting = false, dragging = false;
    POINT previous{}; // last mouse position while orbiting
    V3 anchor{};      // the picked point; the tool moves on the camera-facing plane through it
    float fps = 0;

    App() = default;
    App(const App &) = delete;
    App &operator=(const App &) = delete;
    ~App();

    void create(HINSTANCE instance);
    void createControls();
    void resetCamera();
    void readSliders();
    void resize();
    void togglePause();
    void advanceOne();
    void command(Id id, int code);
    bool key(WPARAM pressed);
    /// Dispatches pending window messages; returns false once the application should exit.
    bool pump();
    int canvasWidth() const;
    int canvasHeight() const;
    bool inCanvas(int x, int y) const;
    V3 ray(int x, int y) const;
    void beginDrag(int x, int y);
    void mouse(int x, int y);
    void stopInteraction();
    void draw();
    void updateStatus();
};
} // namespace material
