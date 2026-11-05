// Entry point: window setup and the fixed-timestep main loop.
#include "App.hpp"
#include <commctrl.h>
#include <algorithm>
#include <chrono>
#include <exception>

namespace {
using namespace material;
using Clock = std::chrono::steady_clock;

constexpr double tick = 1.0 / tickRate; // real seconds per nominal simulation tick (20 substeps)
constexpr auto statusInterval = std::chrono::milliseconds(500);

int run(App &app) {
    auto last = Clock::now(), report = last;
    double backlog = 0; // real seconds not yet simulated, capped so a slow GPU never bursts
    int framesSinceReport = 0;
    while (app.pump()) {
        const auto now = Clock::now();
        backlog = std::min(backlog + std::chrono::duration<double>(now - last).count(), 2 * tick);
        last = now;
        if (IsIconic(app.window)) {
            Sleep(30);
            backlog = 0;
            continue;
        }
        if (app.single) {
            app.advanceOne();
            backlog = 0;
        } else if (app.paused) {
            backlog = 0;
        } else if (backlog >= tick) {
            app.sim->tick(); // one tick per frame at most: slow hardware slows simulated time
            backlog -= tick;
        }
        app.draw();
        ++framesSinceReport;
        SwapBuffers(app.gpu.dc);
        if (now - report > statusInterval) {
            app.fps = static_cast<float>(framesSinceReport / std::chrono::duration<double>(now - report).count());
            framesSinceReport = 0;
            report = now;
            app.updateStatus();
        }
        if (!app.gpu.vsync)
            Sleep(1); // without vertical sync, avoid spinning a core at full speed
    }
    return 0;
}
} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    try {
        SetProcessDPIAware();
        INITCOMMONCONTROLSEX controls{sizeof(controls), ICC_BAR_CLASSES};
        InitCommonControlsEx(&controls);
        App app;
        app.create(instance);
        ShowWindow(app.window, show);
        return run(app);
    } catch (const std::exception &error) {
        MessageBoxA(nullptr, error.what(), "Material Lab 3D", MB_ICONERROR);
        return 1;
    }
}
