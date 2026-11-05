#include "App.hpp"
#include "ModernUI.hpp"
#include <commctrl.h>
#include <windowsx.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace material {
namespace {
constexpr int margin = 20, contentWidth = 280;      // panel content spans x = 20..300
constexpr int buttonWidth = 130, buttonHeight = 32; // two buttons per row: 20..150 and 170..300
constexpr float orbitSpeed = .007f, zoomSpeed = .001f, minDistance = .65f, maxDistance = 4.f;
constexpr float minPitch = -.12f, topPitch = 1.35f; // slightly below the floor up to almost overhead
constexpr float brushMargin = .03f;                 // the tool stays inside the unit cube
constexpr std::array sideOptions{20, 24, 32};       // particles per axis offered by the combo

int comboIndex(Id id) {
    return std::to_underlying(id) - std::to_underlying(Id::material);
}

LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM w, LPARAM l) {
    auto *app = reinterpret_cast<App *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        app = static_cast<App *>(reinterpret_cast<CREATESTRUCTW *>(l)->lpCreateParams);
        app->window = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
    }
    if (message == WM_GETMINMAXINFO) {
        reinterpret_cast<MINMAXINFO *>(l)->ptMinTrackSize = {minimumWidth, minimumHeight};
        return 0;
    }
    if (!app)
        return DefWindowProcW(hwnd, message, w, l);
    const int x = GET_X_LPARAM(l), y = GET_Y_LPARAM(l);
    switch (message) {
    case WM_CLOSE:
        app->running = false;
        return 0;
    case WM_SIZE:
        app->resize();
        return 0;
    case WM_ERASEBKGND:
        return 1; // the sidebar and footer are painted by the themed root, the viewport by OpenGL
    case WM_COMMAND:
        app->command(static_cast<Id>(LOWORD(w)), HIWORD(w));
        return 0;
    case WM_HSCROLL:
        app->readSliders();
        return 0;
    case WM_LBUTTONDOWN:
        app->beginDrag(x, y);
        return 0;
    case WM_RBUTTONDOWN:
        if (app->inCanvas(x, y)) {
            app->orbiting = true;
            app->previous = {x, y};
            SetCapture(hwnd);
        }
        return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_CAPTURECHANGED:
        app->stopInteraction();
        return 0;
    case WM_MOUSEMOVE:
        app->mouse(x, y);
        return 0;
    case WM_MOUSEWHEEL:
        app->camera.distance = std::clamp(app->camera.distance * std::exp(-GET_WHEEL_DELTA_WPARAM(w) * zoomSpeed),
                                          minDistance, maxDistance);
        return 0;
    }
    return DefWindowProcW(hwnd, message, w, l);
}
} // namespace

const std::array<SliderSpec, sliderCount> sliderSpecs{{
    {L"Young's modulus", 2000, 29000, 37, L"{:.0f}", &Settings::young},
    {L"Gravity", 0, 5, 40, L"{:.2f}", &Settings::gravity},
    {L"Contact friction", 0, 1, 35, L"{:.2f}", &Settings::friction},
    {L"Tool radius", .03f, .18f, 40, L"{:.3f}", &Settings::brushRadius},
}};

App::~App() {
    sim.reset();    // GPU objects need the context
    gpu.shutdown(); // the context needs the viewport's DC
    if (window)
        DestroyWindow(window);
}

void App::create(HINSTANCE instance) {
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
    wc.lpszClassName = L"Material3DGPU";
    if (!RegisterClassExW(&wc))
        throw std::runtime_error("Window registration failed.");
    if (!CreateWindowExW(0, wc.lpszClassName, appTitle, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT,
                         CW_USEDEFAULT, initialWidth, initialHeight, nullptr, nullptr, instance, this))
        throw std::runtime_error("Window creation failed.");
    createControls();
    modern::apply(window, {.sidebarWidth = panelWidth,
                           .designWidth = panelWidth,
                           .title = appTitle,
                           .subtitle = appSubtitle,
                           .primaryButtonId = std::to_underlying(Id::pause),
                           .footerHeight = statusHeight});
    // OpenGL renders into its own child window; the root keeps the mouse and keyboard handling.
    viewport = modern::createViewport(window);
    gpu.initialize(viewport, /*depthBuffer=*/true);
    sim = std::make_unique<Simulation>(gpu);
    readSliders();
    resetCamera();
    resize();
}

void App::createControls() {
    auto control = [&](const wchar_t *kind, const wchar_t *text, DWORD style, int x, int y, int w, int h,
                       int id = 0) {
        return CreateWindowExW(0, kind, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, window,
                               reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), GetModuleHandleW(nullptr),
                               nullptr);
    };
    auto combo = [&](const wchar_t *label, int y, Id id, std::initializer_list<const wchar_t *> items,
                     int initial) {
        control(L"STATIC", label, 0, margin, y, contentWidth, 21);
        HWND c = control(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, margin, y + 23, contentWidth, 180,
                         std::to_underlying(id));
        for (auto item : items)
            SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item));
        SendMessageW(c, CB_SETCURSEL, static_cast<WPARAM>(initial), 0);
        return c;
    };
    auto button = [&](const wchar_t *text, int x, int y, int w, Id id) {
        return control(L"BUTTON", text, BS_PUSHBUTTON | WS_TABSTOP, x, y, w, buttonHeight, std::to_underlying(id));
    };

    control(L"STATIC", appTitle, 0, margin, 14, contentWidth, 27);
    control(L"STATIC", appSubtitle, 0, margin, 43, contentWidth, 24);
    combos[comboIndex(Id::material)] =
        combo(L"Material", 76, Id::material, {L"Snow / hardening", L"Sand / Drucker-Prager", L"Clay / plastic yield"},
              0);
    combos[comboIndex(Id::experiment)] = combo(L"Experiment", 139, Id::experiment,
                                               {L"Drop onto sphere", L"Colliding blocks", L"Compression press"}, 0);
    combos[comboIndex(Id::particles)] =
        combo(L"Particles (resets)", 202, Id::particles,
              {L"8,000 / responsive", L"13,824 / balanced", L"32,768 / detailed"}, 1);
    combos[comboIndex(Id::tool)] = combo(L"Left mouse tool", 265, Id::tool,
                                         {L"Grab / spring force", L"Cut / remove material", L"Push / radial force"}, 0);
    combos[comboIndex(Id::color)] =
        combo(L"Particle color", 328, Id::color, {L"Material", L"Speed", L"Elastic volume change"}, 0);
    for (int i = 0; i < sliderCount; ++i) {
        const int y = 395 + i * 61;
        labels[i] = control(L"STATIC", sliderSpecs[i].label, 0, margin, y, contentWidth, 23);
        sliders[i] = control(TRACKBAR_CLASSW, L"", TBS_HORZ | WS_TABSTOP, margin - 6, y + 24, contentWidth + 12, 30,
                             std::to_underlying(Id::slider) + i);
        SendMessageW(sliders[i], TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
        SendMessageW(sliders[i], TBM_SETPOS, TRUE, sliderSpecs[i].initial);
    }
    const int secondColumn = margin + contentWidth - buttonWidth;
    pauseButton = button(L"Pause", margin, 652, buttonWidth, Id::pause);
    button(L"Reset", secondColumn, 652, buttonWidth, Id::reset);
    button(L"Step", margin, 694, buttonWidth, Id::step);
    button(L"Reset camera", secondColumn, 694, buttonWidth, Id::resetCamera);
    control(L"STATIC", L"LMB: use tool on material\nRMB: orbit   Wheel: zoom\nSpace: pause   R: reset   N: step", 0,
            margin, 747, contentWidth, 70);
    status = control(L"STATIC", L"Initializing...", 0, panelWidth + 12, height - 58, width - panelWidth - 24, 54);
}

void App::resetCamera() {
    camera = Camera{.yaw = .65f, .pitch = .35f, .distance = 1.85f, .target = {.5f, .45f, .5f}};
}

void App::readSliders() {
    if (!sim)
        return;
    auto &s = sim->settings;
    for (int i = 0; i < sliderCount; ++i) {
        const auto &spec = sliderSpecs[i];
        const int position = static_cast<int>(SendMessageW(sliders[i], TBM_GETPOS, 0, 0));
        float value = spec.low + (spec.high - spec.low) * (static_cast<float>(position) / 100);
        s.*spec.target = value;
        const auto text = std::format(L"{}: {}", spec.label, std::vformat(spec.format, std::make_wformat_args(value)));
        SetWindowTextW(labels[i], text.c_str());
    }
}

void App::resize() {
    RECT r{};
    GetClientRect(window, &r);
    width = r.right;
    height = r.bottom;
    if (status)
        MoveWindow(status, panelWidth + 12, height - 58, std::max(1, width - panelWidth - 24), 54, TRUE);
}

void App::togglePause() {
    paused = !paused;
    SetWindowTextW(pauseButton, paused ? L"Resume" : L"Pause");
}

void App::advanceOne() {
    sim->tick(1); // a single 1/1200 s substep, for inspecting contacts frame by frame
    single = false;
}

void App::command(Id id, int code) {
    if (!sim)
        return;
    auto &s = sim->settings;
    switch (id) {
    case Id::material:
    case Id::experiment:
    case Id::particles:
    case Id::tool:
    case Id::color: {
        if (code != CBN_SELCHANGE)
            return;
        const int v = static_cast<int>(SendMessageW(combos[static_cast<size_t>(comboIndex(id))], CB_GETCURSEL, 0, 0));
        if (v < 0)
            return;
        switch (id) {
        case Id::material:
            s.material = static_cast<Material>(v);
            break;
        case Id::experiment:
            s.scene = static_cast<Scene>(v);
            break;
        case Id::particles:
            s.side = sideOptions[static_cast<size_t>(v)];
            break;
        case Id::tool:
            s.tool = static_cast<Tool>(v);
            return;
        case Id::color:
            s.color = static_cast<ColorMode>(v);
            return;
        default:
            return;
        }
        stopInteraction();
        sim->reset();
        return;
    }
    case Id::pause:
        togglePause();
        return;
    case Id::reset:
        stopInteraction();
        sim->reset();
        return;
    case Id::step:
        paused = true;
        single = true;
        SetWindowTextW(pauseButton, L"Resume");
        return;
    case Id::resetCamera:
        stopInteraction();
        resetCamera();
        return;
    default:
        return;
    }
}

bool App::key(WPARAM pressed) {
    switch (pressed) {
    case VK_ESCAPE:
        running = false;
        return true;
    case VK_SPACE:
        togglePause();
        return true;
    case 'R':
        command(Id::reset, 0);
        return true;
    case 'N':
        command(Id::step, 0);
        return true;
    default:
        return false;
    }
}

bool App::pump() {
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT)
            running = false;
        if (msg.message == WM_KEYDOWN && msg.hwnd == window && key(msg.wParam))
            continue; // simulation shortcuts apply while the canvas has focus
        if (IsDialogMessageW(window, &msg))
            continue; // Tab navigation between the panel controls
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return running;
}

int App::canvasWidth() const {
    return std::max(1, width - panelWidth);
}

int App::canvasHeight() const {
    return std::max(1, height - statusHeight);
}

bool App::inCanvas(int x, int y) const {
    return x > panelWidth && x < width && y >= 0 && y < height - statusHeight;
}

V3 App::ray(int x, int y) const {
    const float w = static_cast<float>(canvasWidth()), h = static_cast<float>(canvasHeight());
    return camera.ray(2 * static_cast<float>(x - panelWidth) / w - 1, 1 - 2 * static_cast<float>(y) / h, w / h);
}

/// Picks the particle under the cursor; the tool then follows the cursor on the plane facing the camera.
void App::beginDrag(int x, int y) {
    if (!sim || !inCanvas(x, y))
        return;
    const V3 direction = ray(x, y);
    const float depth = sim->pick(camera.eye(), direction);
    if (!std::isfinite(depth))
        return;
    anchor = camera.eye() + direction * depth;
    sim->settings.brush = anchor;
    sim->settings.dragging = true;
    dragging = true;
    SetCapture(window);
}

void App::mouse(int x, int y) {
    if (orbiting) {
        camera.yaw += static_cast<float>(x - previous.x) * orbitSpeed;
        camera.pitch = std::clamp(camera.pitch + static_cast<float>(y - previous.y) * orbitSpeed, minPitch, topPitch);
    }
    if (dragging && sim) {
        const V3 direction = ray(x, y), normal = camera.forward(), eye = camera.eye();
        const float denominator = direction.dot(normal);
        if (std::abs(denominator) > .01f) {
            const V3 p = eye + direction * ((anchor - eye).dot(normal) / denominator);
            sim->settings.brush = {std::clamp(p.x, brushMargin, 1 - brushMargin),
                                   std::clamp(p.y, brushMargin, 1 - brushMargin),
                                   std::clamp(p.z, brushMargin, 1 - brushMargin)};
        }
    }
    previous = {x, y};
}

void App::stopInteraction() {
    dragging = false;
    orbiting = false;
    if (sim)
        sim->settings.dragging = false;
    if (GetCapture() == window)
        ReleaseCapture();
}

void App::draw() {
    const int w = canvasWidth(), h = canvasHeight();
    glViewport(0, 0, w, h);
    glClearColor(.012f, .018f, .032f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    sim->draw(camera, w, h);
}

void App::updateStatus() {
    sim->measure();
    const auto &d = sim->diagnostics;
    const bool healthy = sim->healthy();
    if (!healthy) {
        paused = true;
        SetWindowTextW(pauseButton, L"Resume");
    }
    const auto text = std::format(L"{} / {} particles | {}³ grid | GPU {:.2f} ms | {:.0f} fps | time {:.2f} s\n"
                                  L"Max speed {:.2f} | invalid {} | speed clamps {} | boundary corrections {}{}",
                                  d.active(), sim->count(), gridSize, sim->gpuMs, fps, sim->time, d.maxSpeed(),
                                  d.invalid(), d.speedClamps(), d.boundaryCorrections(),
                                  healthy ? L"" : L" | INVALID STATE: reset required");
    SetWindowTextW(status, text.c_str());
}
} // namespace material
