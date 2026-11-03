#include "ModernUI.hpp"
#include "GpuApi.hpp"
#include "Math3D.hpp"
#include "Shaders.hpp"
#include <commctrl.h>
#include <windowsx.h>
#include <bit>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>

namespace material {
constexpr int panel = 320, gridSize = 40;
constexpr float stepSize = 1.f / 1200;

struct alignas(16) Particle {
    std::array<float, 4> x, v, f0, f1, f2, c0, c1, c2;
};

static_assert(sizeof(Particle) == 128);

struct Settings {
    int material{}, scene{}, side = 24, tool{}, color{};
    float young = 12000, gravity = 2, friction = .35f, brushRadius = .09f;
    bool dragging{};
    V3 brush{};
};

class Simulator {
    GL &gl;
    GLuint init{}, clear{}, p2g{}, grid{}, g2p{}, pickProgram{}, particleProgram{}, sceneProgram{};
    GLuint particleBuffer{}, accumulatorBuffer{}, velocityBuffer{}, statsBuffer{}, pickBuffer{}, timer{};
    bool timerPending{};
    void common(GLuint program) {
        gl.UseProgram(program);
        gl.integer(program, "count", count());
        gl.integer(program, "resolution", gridSize);
        gl.integer(program, "side", settings.side);
        gl.integer(program, "materialType", settings.material);
        gl.integer(program, "scene", settings.scene);
        gl.integer(program, "tool", settings.tool);
        gl.integer(program, "dragging", settings.dragging ? 1 : 0);
        gl.integer(program, "collect", 1);
        gl.scalar(program, "dt", stepSize);
        gl.scalar(program, "young", settings.young);
        gl.scalar(program, "gravity", settings.gravity);
        gl.scalar(program, "friction", settings.friction);
        gl.scalar(program, "brushRadius", settings.brushRadius);
        gl.vector(program, "brush", settings.brush.x, settings.brush.y, settings.brush.z);
    }

  public:
    Settings settings;
    float time{}, gpuMilliseconds{};
    std::array<unsigned, 16> diagnostics{};
    explicit Simulator(GL &api) : gl(api) {
        auto compute = [&](const std::string &source, const char *name) {
            return gl.program({{COMPUTE, shaders::common + source}}, name);
        };
        init = compute(shaders::initialize, "Initialize");
        clear = compute(shaders::clear, "Clear grid");
        p2g = compute(shaders::p2g, "Particle to grid");
        grid = compute(shaders::updateGrid, "Update grid");
        g2p = compute(shaders::g2p, "Grid to particle");
        pickProgram = compute(shaders::pick, "GPU picking");
        particleProgram = gl.program(
            {{VERTEX, shaders::common + shaders::particleVertex}, {FRAGMENT, shaders::particleFragment}},
            "Particle spheres");
        sceneProgram =
            gl.program({{VERTEX, shaders::sceneVertex}, {FRAGMENT, shaders::sceneFragment}}, "Scene");
        particleBuffer = gl.buffer(0, 32 * 32 * 32 * sizeof(Particle));
        accumulatorBuffer = gl.buffer(1, gridSize * gridSize * gridSize * 16);
        velocityBuffer = gl.buffer(2, gridSize * gridSize * gridSize * 16);
        statsBuffer = gl.buffer(3, sizeof(diagnostics));
        pickBuffer = gl.buffer(4, 4);
        gl.GenQueries(1, &timer);
        reset();
    }
    ~Simulator() {
        gl.DeleteQueries(1, &timer);
    }
    int count() const {
        return settings.side * settings.side * settings.side;
    }
    void reset() {
        time = 0;
        settings.dragging = false;
        common(init);
        gl.dispatch(count());
        common(clear);
        gl.dispatch(gridSize * gridSize * gridSize);
        diagnostics = {};
        diagnostics[0] = count();
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.BindBuffer(SSBO, statsBuffer);
        gl.BufferSubData(SSBO, 0, sizeof(diagnostics), diagnostics.data());
    }
    void advance(int substeps = 20) {
        if (timerPending) {
            GLuint ready{};
            gl.GetQueryObjectuiv(timer, 0x8867, &ready);
            if (ready) {
                unsigned long long ns{};
                gl.GetQueryObjectui64v(timer, 0x8866, &ns);
                gpuMilliseconds = static_cast<float>(ns) * 1e-6f;
                timerPending = false;
            }
        }
        bool measure = !timerPending;
        if (measure)
            gl.BeginQuery(0x88BF, timer);
        for (GLuint p : {clear, p2g, grid, g2p})
            common(p);
        for (int i = 0; i < substeps; ++i) {
            gl.UseProgram(clear);
            gl.dispatch(gridSize * gridSize * gridSize);
            gl.UseProgram(p2g);
            gl.dispatch(count());
            gl.UseProgram(grid);
            gl.scalar(grid, "time", time);
            gl.dispatch(gridSize * gridSize * gridSize);
            gl.UseProgram(g2p);
            gl.dispatch(count());
            time += stepSize;
        }
        if (measure) {
            gl.EndQuery(0x88BF);
            timerPending = true;
        }
    }
    void readDiagnostics() {
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.BindBuffer(SSBO, statsBuffer);
        gl.GetBufferSubData(SSBO, 0, sizeof(diagnostics), diagnostics.data());
    }
    std::vector<Particle> snapshot() {
        std::vector<Particle> p(count());
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.BindBuffer(SSBO, particleBuffer);
        gl.GetBufferSubData(SSBO, 0, static_cast<std::ptrdiff_t>(p.size() * sizeof(Particle)), p.data());
        return p;
    }
    double gridMass() {
        std::vector<float> g(gridSize * gridSize * gridSize * 4);
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.BindBuffer(SSBO, accumulatorBuffer);
        gl.GetBufferSubData(SSBO, 0, static_cast<std::ptrdiff_t>(g.size() * sizeof(float)), g.data());
        double mass = 0;
        for (size_t i = 3; i < g.size(); i += 4)
            mass += g[i];
        return mass;
    }
    float pick(V3 origin, V3 direction) {
        unsigned value = std::bit_cast<unsigned>(std::numeric_limits<float>::infinity());
        gl.BindBuffer(SSBO, pickBuffer);
        gl.BufferSubData(SSBO, 0, 4, &value);
        common(pickProgram);
        gl.vector(pickProgram, "rayOrigin", origin.x, origin.y, origin.z);
        gl.vector(pickProgram, "rayDirection", direction.x, direction.y, direction.z);
        gl.dispatch(count());
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.GetBufferSubData(SSBO, 0, 4, &value);
        return std::bit_cast<float>(value);
    }
    void draw(const Camera &camera, int width, int height) {
        auto view = camera.view(), projection = perspective(static_cast<float>(width) / std::max(1, height));
        gl.BindVertexArray(gl.vao);
        glEnable(GL_DEPTH_TEST);
        gl.UseProgram(sceneProgram);
        gl.matrix(sceneProgram, "view", view.v.data());
        gl.matrix(sceneProgram, "projection", projection.v.data());
        gl.integer(sceneProgram, "shape", 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        if (settings.scene == 0) {
            gl.integer(sceneProgram, "shape", 1);
            glDrawArrays(GL_TRIANGLES, 0, 48 * 24 * 6);
        }
        if (settings.scene == 2) {
            gl.integer(sceneProgram, "shape", 2);
            gl.scalar(sceneProgram, "press", .84f - .60f * (.5f - .5f * std::cos(time * 1.4f)));
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        common(particleProgram);
        gl.matrix(particleProgram, "view", view.v.data());
        gl.matrix(particleProgram, "projection", projection.v.data());
        gl.integer(particleProgram, "colorMode", settings.color);
        gl.DrawArraysInstanced(GL_TRIANGLES, 0, 6, count());
    }
};
void capture(GL &, const std::filesystem::path &file, int x, int y, int w, int h) {
    int stride = (w * 3 + 3) & ~3;
    std::vector<unsigned char> data(static_cast<size_t>(stride) * h);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(x, y, w, h, GL_BGR_EXT, GL_UNSIGNED_BYTE, data.data());
    BITMAPFILEHEADER header{};
    BITMAPINFOHEADER info{};
    header.bfType = 0x4d42;
    header.bfOffBits = sizeof(header) + sizeof(info);
    header.bfSize = header.bfOffBits + static_cast<DWORD>(data.size());
    info.biSize = sizeof(info);
    info.biWidth = w;
    info.biHeight = h;
    info.biPlanes = 1;
    info.biBitCount = 24;
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<char *>(&header), sizeof(header));
    out.write(reinterpret_cast<char *>(&info), sizeof(info));
    out.write(reinterpret_cast<char *>(data.data()), data.size());
    if (!out)
        throw std::runtime_error("Framebuffer capture failed.");
}

struct Application {
    GL gl;
    std::unique_ptr<Simulator> sim;
    Camera camera;
    HWND window{}, status{}, pauseButton{}, combos[5]{}, sliders[4]{}, labels[4]{};
    HFONT font{};
    bool running = true, paused{}, single{}, orbit{}, drag{};
    POINT previous{};
    V3 dragPlane{};
    int width = 1200, height = 850;
    float fps{};
    ~Application() {
        sim.reset();
        if (font)
            DeleteObject(font);
    }
    HWND control(const wchar_t *type, const std::wstring &text, DWORD style, int x, int y, int w, int h,
                 int id = 0) {
        HWND item = CreateWindowExW(0, type, text.c_str(), WS_CHILD | WS_VISIBLE | style, x, y, w, h, window,
                                    reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                    GetModuleHandleW(nullptr), nullptr);
        SendMessageW(item, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return item;
    }
    HWND combo(const wchar_t *title, int y, int id, std::initializer_list<const wchar_t *> items) {
        control(L"STATIC", title, 0, 18, y, 248, 22);
        auto c = control(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, 18, y + 24, 248, 180, id);
        for (auto item : items)
            SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item));
        SendMessageW(c, CB_SETCURSEL, 0, 0);
        return c;
    }
    void controls() {
        font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                           CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        control(L"STATIC", L"MATERIAL LAB / 3D GPU", 0, 18, 15, 250, 27);
        control(L"STATIC", L"OpenGL 4.3 compute / MLS-MPM", 0, 18, 44, 250, 24);
        combos[0] = combo(L"Material", 80, 10,
                          {L"Snow / hardening", L"Sand / Drucker-Prager", L"Clay / plastic yield"});
        combos[1] =
            combo(L"Experiment", 146, 11, {L"Drop onto sphere", L"Colliding blocks", L"Compression press"});
        combos[2] = combo(L"Particles (resets simulation)", 212, 12,
                          {L"8,000 / responsive", L"13,824 / balanced", L"32,768 / detailed"});
        SendMessageW(combos[2], CB_SETCURSEL, 1, 0);
        combos[3] = combo(L"Mouse tool", 278, 13,
                          {L"Grab / spring force", L"Cut / remove material", L"Push / radial force"});
        combos[4] = combo(L"Particle color", 344, 14, {L"Material", L"Speed", L"Elastic volume change"});
        const wchar_t *titles[] = {L"Stiffness", L"Gravity", L"Friction", L"Tool radius"};
        int positions[] = {37, 40, 35, 40};
        for (int i = 0; i < 4; ++i) {
            labels[i] = control(L"STATIC", titles[i], 0, 18, 412 + i * 64, 250, 22);
            sliders[i] =
                control(TRACKBAR_CLASSW, L"", TBS_HORZ | WS_TABSTOP, 12, 436 + i * 64, 260, 30, 20 + i);
            SendMessageW(sliders[i], TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
            SendMessageW(sliders[i], TBM_SETPOS, TRUE, positions[i]);
        }
        pauseButton = control(L"BUTTON", L"Pause", BS_PUSHBUTTON | WS_TABSTOP, 18, 678, 116, 32, 30);
        control(L"BUTTON", L"Reset", BS_PUSHBUTTON | WS_TABSTOP, 146, 678, 120, 32, 31);
        control(L"BUTTON", L"Step", BS_PUSHBUTTON | WS_TABSTOP, 18, 718, 116, 32, 32);
        control(L"BUTTON", L"Reset camera", BS_PUSHBUTTON | WS_TABSTOP, 146, 718, 120, 32, 33);
        control(L"STATIC",
                L"LMB: use tool on material\nRMB: orbit   Wheel: zoom\nSpace: pause   R: reset   N: step", 0,
                18, 768, 250, 70);
        status = control(L"STATIC", L"Initializing compute shaders...", 0, panel + 12, height - 58,
                         width - panel - 24, 54);
        updateSliders();
    }
    void updateSliders() {
        if (!sim)
            return;
        int v[4]{};
        for (int i = 0; i < 4; ++i)
            v[i] = static_cast<int>(SendMessageW(sliders[i], TBM_GETPOS, 0, 0));
        sim->settings.young = 2000 + v[0] * 270.f;
        sim->settings.gravity = v[1] * .05f;
        sim->settings.friction = v[2] * .01f;
        sim->settings.brushRadius = .03f + v[3] * .0015f;
        std::wstring text[] = {std::format(L"Young's modulus: {:.0f}", sim->settings.young),
                               std::format(L"Gravity: {:.2f}", sim->settings.gravity),
                               std::format(L"Contact friction: {:.2f}", sim->settings.friction),
                               std::format(L"Tool radius: {:.3f}", sim->settings.brushRadius)};
        for (int i = 0; i < 4; ++i)
            SetWindowTextW(labels[i], text[i].c_str());
    }
    void resize() {
        RECT rect{};
        GetClientRect(window, &rect);
        width = rect.right;
        height = rect.bottom;
        if (status)
            MoveWindow(status, panel + 12, height - 58, std::max(1, width - panel - 24), 54, TRUE);
    }
    void toggle() {
        paused = !paused;
        SetWindowTextW(pauseButton, paused ? L"Resume" : L"Pause");
    }
    void command(int id, int notification) {
        if (!sim)
            return;
        if (id >= 10 && id <= 14 && notification == CBN_SELCHANGE) {
            int value = static_cast<int>(SendMessageW(combos[id - 10], CB_GETCURSEL, 0, 0));
            if (id == 10)
                sim->settings.material = value;
            if (id == 11)
                sim->settings.scene = value;
            if (id == 12)
                sim->settings.side = std::array{20, 24, 32}[value];
            if (id == 13)
                sim->settings.tool = value;
            if (id == 14)
                sim->settings.color = value;
            if (id <= 12)
                sim->reset();
        }
        if (id == 30)
            toggle();
        if (id == 31)
            sim->reset();
        if (id == 32) {
            paused = true;
            single = true;
            SetWindowTextW(pauseButton, L"Resume");
        }
        if (id == 33)
            camera = Camera{};
    }
    bool inCanvas(int x, int y) const {
        return x > panel && x < width && y >= 0 && y < height - 64;
    }
    V3 ray(int x, int y) const {
        float w = static_cast<float>(std::max(1, width - panel)),
              h = static_cast<float>(std::max(1, height - 64));
        return camera.ray(2 * (x - panel) / w - 1, 1 - 2 * y / h, w / h);
    }
    void beginDrag(int x, int y) {
        if (!sim || !inCanvas(x, y))
            return;
        auto direction = ray(x, y);
        float depth = sim->pick(camera.eye(), direction);
        if (std::isfinite(depth)) {
            dragPlane = camera.eye() + direction * depth;
            sim->settings.brush = dragPlane;
            drag = true;
            sim->settings.dragging = true;
            SetCapture(window);
        }
    }
    void moveMouse(int x, int y) {
        if (orbit) {
            camera.yaw += (x - previous.x) * .007f;
            camera.pitch = std::clamp(camera.pitch + (y - previous.y) * .007f, -.12f, 1.35f);
        }
        if (drag && sim) {
            auto direction = ray(x, y), normal = camera.forward();
            float denominator = direction.dot(normal);
            if (std::abs(denominator) > .01f) {
                float t = (dragPlane - camera.eye()).dot(normal) / denominator;
                V3 p = camera.eye() + direction * t;
                sim->settings.brush = {std::clamp(p.x, .03f, .97f), std::clamp(p.y, .03f, .97f),
                                       std::clamp(p.z, .03f, .97f)};
            }
        }
        previous = {x, y};
    }
    void stopDrag() {
        drag = false;
        orbit = false;
        if (sim)
            sim->settings.dragging = false;
        ReleaseCapture();
    }
    void draw() {
        int w = std::max(1, width - panel), h = std::max(1, height - 64);
        glViewport(0, 0, width, height);
        glClearColor(.012f, .018f, .032f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, 0, panel, height);
        glClearColor(246.f / 255, 248.f / 255, 252.f / 255, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glViewport(panel, 64, w, h);
        sim->draw(camera, w, h);
    }
    void updateStatus() {
        sim->readDiagnostics();
        auto d = sim->diagnostics;
        if (d[1]) {
            paused = true;
            SetWindowTextW(pauseButton, L"Resume");
        }
        auto message =
            std::format(L"{} / {} particles  |  40³ grid  |  GPU {:.2f} ms  |  {:.0f} fps\nTime {:.2f} s  |  "
                        L"max speed {:.2f}  |  invalid {}  |  speed clamps {}  |  boundary corrections {}{}",
                        d[0], sim->count(), sim->gpuMilliseconds, fps, sim->time, std::bit_cast<float>(d[4]),
                        d[1], d[2], d[3], d[1] ? L"  PAUSED: reset required" : L"");
        SetWindowTextW(status, message.c_str());
    }
};
LRESULT CALLBACK procedure(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
    auto app = reinterpret_cast<Application *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (msg == WM_NCCREATE) {
        app = static_cast<Application *>(reinterpret_cast<CREATESTRUCTW *>(l)->lpCreateParams);
        app->window = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
    }
    if (app) {
        switch (msg) {
        case WM_CLOSE:
            app->running = false;
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_SIZE:
            app->resize();
            return 0;
        case WM_GETMINMAXINFO:
            reinterpret_cast<MINMAXINFO *>(l)->ptMinTrackSize = {1080, 760};
            return 0;
        case WM_COMMAND:
            app->command(LOWORD(w), HIWORD(w));
            return 0;
        case WM_HSCROLL:
            app->updateSliders();
            return 0;
        case WM_LBUTTONDOWN:
            app->beginDrag(GET_X_LPARAM(l), GET_Y_LPARAM(l));
            return 0;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_CAPTURECHANGED:
            app->stopDrag();
            return 0;
        case WM_RBUTTONDOWN:
            if (app->inCanvas(GET_X_LPARAM(l), GET_Y_LPARAM(l))) {
                app->orbit = true;
                app->previous = {GET_X_LPARAM(l), GET_Y_LPARAM(l)};
                SetCapture(hwnd);
            }
            return 0;
        case WM_MOUSEMOVE:
            app->moveMouse(GET_X_LPARAM(l), GET_Y_LPARAM(l));
            return 0;
        case WM_MOUSEWHEEL:
            app->camera.distance =
                std::clamp(app->camera.distance * std::exp(-GET_WHEEL_DELTA_WPARAM(w) * .001f), .65f, 4.f);
            return 0;
        case WM_CTLCOLORSTATIC:
            SetBkColor(reinterpret_cast<HDC>(w), RGB(240, 242, 246));
            SetTextColor(reinterpret_cast<HDC>(w), RGB(30, 40, 55));
            return reinterpret_cast<LRESULT>(GetSysColorBrush(COLOR_BTNFACE));
        }
    }
    return DefWindowProcW(hwnd, msg, w, l);
}

bool validate(Simulator &sim, std::ostream &out, const char *label, bool massCheck = true) {
    auto data = sim.snapshot();
    sim.readDiagnostics();
    double y = 0;
    int active = 0;
    bool finite = true;
    for (const auto &p : data) {
        if (p.x[3] == 0)
            continue;
        ++active;
        y += p.x[1];
        const auto values = std::bit_cast<std::array<float, 32>>(p);
        for (int j = 0; j < 32; ++j)
            finite = finite && std::isfinite(values[j]);
        for (int j = 0; j < 3; ++j)
            finite = finite && p.x[j] >= 0 && p.x[j] <= 1;
    }
    double particleMass = 1000 * std::pow(.36 / sim.settings.side, 3) * active, gridMass = sim.gridMass();
    double massError = std::abs(particleMass - gridMass) / std::max(1e-9, particleMass);
    bool ok = finite && sim.diagnostics[1] == 0 && sim.diagnostics[2] == 0 && active > 0 &&
              (!massCheck || massError < 2e-5);
    out << label << ": " << (ok ? "PASS" : "FAIL") << " active=" << active
        << " centroidY=" << y / std::max(1, active) << " gridMassError=" << massError
        << " invalid=" << sim.diagnostics[1] << " clamps=" << sim.diagnostics[2] << "\n";
    return ok;
}
bool tests(Application &app, const std::filesystem::path &folder) {
    auto &sim = *app.sim;
    std::ofstream out(folder / "Material3D-tests.txt");
    out << std::unitbuf << app.gl.driver << "\n";
    bool ok = true;
    for (int material = 0; material < 3; ++material) {
        for (int scene = 0; scene < 3; ++scene) {
            sim.settings = Settings{};
            sim.settings.material = material;
            sim.settings.scene = scene;
            sim.reset();
            for (int frame = 0; frame < 120; ++frame)
                sim.advance();
            ok = validate(sim, out, std::format("Material {} scene {}", material, scene).c_str()) && ok;
            if (scene == 0) {
                auto falling = sim.snapshot();
                double meanHeight = 0;
                for (const auto &particle : falling)
                    meanHeight += particle.x[1];
                const bool fell = meanHeight / falling.size() < .4;
                out << "Gravity response: " << (fell ? "PASS" : "FAIL") << "\n";
                ok = fell && ok;
                app.draw();
                capture(app.gl, folder / std::format("Material3D-{}.bmp", material), panel, 64,
                        app.width - panel, app.height - 64);
                SwapBuffers(app.gl.dc);
            }
        }
    }
    // Exercise both particle-count extremes and the stiffest, highest-gravity setting.
    for (int side : {20, 32}) {
        for (int material = 0; material < 3; ++material) {
            sim.settings = Settings{};
            sim.settings.side = side;
            sim.settings.material = material;
            sim.settings.scene = 1;
            sim.settings.young = 29000;
            sim.settings.gravity = 5;
            sim.settings.friction = 1;
            sim.reset();
            for (int frame = 0; frame < 90; ++frame)
                sim.advance();
            ok = validate(sim, out, std::format("Extreme side {} material {}", side, material).c_str()) && ok;
        }
    }
    sim.settings = Settings{};
    sim.settings.scene = 1;
    sim.settings.gravity = 0;
    sim.reset();
    auto before = sim.snapshot();
    for (int i = 0; i < 30; ++i)
        sim.advance();
    auto after = sim.snapshot();
    double meanBefore = 0, meanAfter = 0;
    for (size_t i = 0; i < before.size(); ++i) {
        meanBefore += before[i].x[1];
        meanAfter += after[i].x[1];
    }
    double drift = std::abs(meanAfter - meanBefore) / before.size();
    bool symmetric = drift < 1e-4;
    out << "Zero gravity centroid symmetry: " << (symmetric ? "PASS" : "FAIL") << " drift=" << drift << "\n";
    ok = ok && symmetric;
    sim.settings = Settings{};
    sim.settings.scene = 2;
    sim.reset();
    sim.settings.tool = 1;
    sim.settings.dragging = true;
    sim.settings.brush = {.5f, .32f, .5f};
    sim.settings.brushRadius = .12f;
    sim.advance(2);
    sim.readDiagnostics();
    bool cut = sim.diagnostics[0] > 0 && sim.diagnostics[0] < static_cast<unsigned>(sim.count());
    out << "GPU cutting: " << (cut ? "PASS" : "FAIL") << " remaining=" << sim.diagnostics[0] << "\n";
    ok = ok && cut;
    sim.settings = Settings{};
    sim.reset();
    float pickDepth = sim.pick({.5f, .7f, 1.5f}, {0, 0, -1});
    bool picked = std::isfinite(pickDepth) && pickDepth > .8f && pickDepth < 1.3f;
    out << "GPU ray picking: " << (picked ? "PASS" : "FAIL") << " depth=" << pickDepth << "\n";
    ok = ok && picked;
    GLenum error = glGetError();
    out << "OpenGL error: " << error << "\n";
    ok = ok && error == GL_NO_ERROR;
    out << (ok ? "ALL PASSED" : "FAILED") << "\n";
    return ok;
}
} // namespace material

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR commandLine, int show) {
    using namespace material;
    std::wstring command = commandLine;
    bool test = command.find(L"--gpu-test") != std::wstring::npos,
        smoke = command.find(L"--smoke") != std::wstring::npos;
    auto folder = std::filesystem::current_path();
    try {
        SetProcessDPIAware();
        INITCOMMONCONTROLSEX cc{ sizeof(cc), ICC_BAR_CLASSES };
        InitCommonControlsEx(&cc);
        Application app;
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = procedure;
        wc.hInstance = instance;
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
        wc.lpszClassName = L"Material3DGPU";
        if (!RegisterClassExW(&wc))
            throw std::runtime_error("Window registration failed.");
        HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"Material Lab / 3D GPU / OpenGL 4.3",
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 1320,
            960, nullptr, nullptr, instance, &app);
        if (!hwnd)
            throw std::runtime_error("Window creation failed.");
        app.gl.initialize(hwnd);
        app.sim = std::make_unique<Simulator>(app.gl);
        app.controls();
        modern::apply(hwnd, panel, 284, L"Material Lab", L"Materials in motion");
        app.resize();
        ShowWindow(hwnd, (smoke || test) ? SW_SHOWNOACTIVATE : show);
        if (test)
            return tests(app, folder) ? 0 : 1;
        using Clock = std::chrono::steady_clock;
        auto last = Clock::now(), statsTime = last;
        double accumulator = 0;
        int frames = 0, counter = 0;
        while (app.running) {
            MSG msg{};
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT)
                    app.running = false;
                if (msg.message == WM_KEYDOWN && msg.hwnd == hwnd) {
                    if (msg.wParam == VK_ESCAPE)
                        app.running = false;
                    if (msg.wParam == VK_SPACE) {
                        app.toggle();
                        continue;
                    }
                    if (msg.wParam == 'R') {
                        app.sim->reset();
                        continue;
                    }
                    if (msg.wParam == 'N') {
                        app.command(32, 0);
                        continue;
                    }
                }
                if (IsDialogMessageW(hwnd, &msg))
                    continue;
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (!app.running)
                break;
            auto now = Clock::now();
            accumulator = std::min(.05, accumulator + std::chrono::duration<double>(now - last).count());
            last = now;
            if (IsIconic(hwnd)) {
                Sleep(30);
                accumulator = 0;
                continue;
            }
            if (smoke)
                app.sim->advance();
            else if (app.single) {
                app.sim->advance(1);
                app.single = false;
                accumulator = 0;
            }
            else if (!app.paused && accumulator >= 1. / 60) {
                app.sim->advance();
                accumulator = 0;
            }
            else if (app.paused)
                accumulator = 0;
            app.draw();
            ++frames;
            ++counter;
            if (smoke && frames == 120) {
                std::ofstream out(folder / "Lab01_smoke.txt");
                out << app.gl.driver << "\n";
                bool ok = validate(*app.sim, out, "3D GPU smoke");
                auto error = glGetError();
                out << "OpenGL error: " << error << "\n";
                capture(app.gl, folder / "Lab01.bmp", panel, 64, app.width - panel, app.height - 64);
                return ok && error == 0 ? 0 : 1;
            }
            SwapBuffers(app.gl.dc);
            if (now - statsTime > std::chrono::milliseconds(500)) {
                app.fps =
                    counter / static_cast<float>(std::chrono::duration<double>(now - statsTime).count());
                counter = 0;
                statsTime = now;
                app.updateStatus();
            }
            Sleep(1);
        }
        return 0;
    }
    catch (const std::exception& e) {
        if (test || smoke) {
            std::ofstream out(folder / "Material3D-error.txt");
            out << e.what();
        }
        else
            MessageBoxA(nullptr, e.what(), "Material Lab 3D", MB_ICONERROR);
        return 1;
    }
}
