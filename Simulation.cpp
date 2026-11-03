#include "Lab.hpp"
#include <format>

namespace physics {
namespace {
// Row-major 2x2 matrix. The simulator uses APIC transfers and fixed-corotated stress.
struct Mat {
    float a = 1, b = 0, c = 0, d = 1;
    static Mat zero() {
        return {0, 0, 0, 0};
    }
    Mat operator+(Mat m) const {
        return {a + m.a, b + m.b, c + m.c, d + m.d};
    }
    Mat operator-(Mat m) const {
        return {a - m.a, b - m.b, c - m.c, d - m.d};
    }
    Mat operator*(float s) const {
        return {a * s, b * s, c * s, d * s};
    }
    Mat operator*(Mat m) const {
        return {a * m.a + b * m.c, a * m.b + b * m.d, c * m.a + d * m.c, c * m.b + d * m.d};
    }
    Vec2 operator*(Vec2 v) const {
        return {a * v.x + b * v.y, c * v.x + d * v.y};
    }
    Mat transpose() const {
        return {a, c, b, d};
    }
    float det() const {
        return a * d - b * c;
    }
};
Mat outer(Vec2 a, Vec2 b) {
    return {a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y};
}
struct Svd {
    Mat u, v;
    Vec2 s;
};
Svd decompose(Mat f) {
    const Mat q = f.transpose() * f;
    float angle = .5f * std::atan2(2 * q.b, q.a - q.d);
    float c = std::cos(angle), s = std::sin(angle);
    Mat v{c, -s, s, c}, fv = f * v;
    Vec2 svals{std::max(1e-6f, Vec2{fv.a, fv.c}.length()), std::max(1e-6f, Vec2{fv.b, fv.d}.length())};
    return {{fv.a / svals.x, fv.b / svals.y, fv.c / svals.x, fv.d / svals.y}, v, svals};
}
class Material final : public Lab {
    static constexpr int n = 64;
    struct Particle {
        Vec2 p, v;
        Mat f{}, c = Mat::zero();
        float plastic = 1;
    };
    struct Node {
        Vec2 v{};
        float mass{};
    };
    std::vector<Particle> particles;
    std::array<Node, n * n> grid{};

  public:
    Material() {
        reset();
    }
    Description describe() const override {
        return {L"01 / Material Lab",
                L"Particle-grid continuum mechanics. Compare elastic and plastic response.",
                {L"Snow", L"Granular material", L"Plastic clay"},
                {L"Stiffness", L"Gravity"},
                L"Left drag: push material\nRight hold: cut / remove particles\nModes reset the experiment."};
    }
    void reset() override {
        particles.clear();
        std::mt19937 rng(42);
        for (int j = 0; j < 36; ++j)
            for (int i = 0; i < 48; ++i)
                particles.push_back(
                    {{.24f + i * .0105f + randomUnit(rng) * .002f, .56f + j * .0105f}, {0, 0}});
    }
    void step(float dt, const Input &input) override {
        // Small explicit substeps keep the elastic wave CFL bound conservative.
        constexpr int steps = 40;
        const float h = dt / steps, dx = 1.f / n;
        if (input.right)
            std::erase_if(particles, [&](const auto &p) { return (p.p - input.mouse).length() < .035f; });
        for (int sub = 0; sub < steps; ++sub) {
            grid.fill({});
            for (auto &p : particles) {
                auto sv = decompose(p.f);
                Mat rotation = sv.u * sv.v.transpose();
                float hard = mode == 0 ? std::exp(std::clamp(6 * (1 - p.plastic), -2.f, 2.f)) : 1.f;
                float young = (2000 + parameters[0] * 8000) * hard, mu = young / 2.6f,
                      lambda = young * .3f / (1.3f * .4f);
                float j = p.f.det();
                Mat stress = ((p.f - rotation) * p.f.transpose()) * (2 * mu) + Mat{} * (lambda * j * (j - 1));
                Mat affine = p.c + stress * (-h); // rest volume .25*dx^2 cancels 4/dx^2; mass = 1
                Vec2 xp = p.p * static_cast<float>(n);
                int bx = static_cast<int>(xp.x - .5f), by = static_cast<int>(xp.y - .5f);
                Vec2 f = xp - Vec2{static_cast<float>(bx), static_cast<float>(by)};
                std::array<Vec2, 3> w{{{.5f * (1.5f - f.x) * (1.5f - f.x), .5f * (1.5f - f.y) * (1.5f - f.y)},
                                       {.75f - (f.x - 1) * (f.x - 1), .75f - (f.y - 1) * (f.y - 1)},
                                       {.5f * (f.x - .5f) * (f.x - .5f), .5f * (f.y - .5f) * (f.y - .5f)}}};
                for (int y = 0; y < 3; ++y)
                    for (int x = 0; x < 3; ++x) {
                        if (bx + x < 0 || bx + x >= n || by + y < 0 || by + y >= n)
                            continue;
                        float weight = w[x].x * w[y].y;
                        Vec2 d = (Vec2{static_cast<float>(x), static_cast<float>(y)} - f) * dx;
                        auto &node = grid[(by + y) * n + bx + x];
                        node.v += (p.v + affine * d) * weight;
                        node.mass += weight;
                    }
            }
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) {
                    auto &node = grid[y * n + x];
                    if (node.mass == 0)
                        continue;
                    node.v = node.v / node.mass;
                    node.v.y -= h * (.4f + parameters[1] * 2);
                    Vec2 position{static_cast<float>(x) / n, static_cast<float>(y) / n};
                    if (input.left && (position - input.mouse).length() < .12f)
                        node.v += (position - input.mouse) * (h * 90);
                    Vec2 delta = position - Vec2{.5f, .32f};
                    float distance = delta.length();
                    if (distance < .105f && distance > 1e-5f) {
                        auto normal = delta / distance;
                        node.v = node.v - normal * std::min(0.f, node.v.dot(normal));
                    }
                    if (x < 3 && node.v.x < 0)
                        node.v.x = 0;
                    if (x > n - 4 && node.v.x > 0)
                        node.v.x = 0;
                    if (y < 3 && node.v.y < 0) {
                        node.v.y = 0;
                        node.v.x *= .92f;
                    }
                    if (y > n - 4 && node.v.y > 0)
                        node.v.y = 0;
                }
            for (auto &p : particles) {
                Vec2 xp = p.p * static_cast<float>(n);
                int bx = static_cast<int>(xp.x - .5f), by = static_cast<int>(xp.y - .5f);
                Vec2 f = xp - Vec2{static_cast<float>(bx), static_cast<float>(by)};
                std::array<Vec2, 3> w{{{.5f * (1.5f - f.x) * (1.5f - f.x), .5f * (1.5f - f.y) * (1.5f - f.y)},
                                       {.75f - (f.x - 1) * (f.x - 1), .75f - (f.y - 1) * (f.y - 1)},
                                       {.5f * (f.x - .5f) * (f.x - .5f), .5f * (f.y - .5f) * (f.y - .5f)}}};
                p.v = {};
                p.c = Mat::zero();
                for (int y = 0; y < 3; ++y)
                    for (int x = 0; x < 3; ++x) {
                        if (bx + x < 0 || bx + x >= n || by + y < 0 || by + y >= n)
                            continue;
                        auto v = grid[(by + y) * n + bx + x].v;
                        float weight = w[x].x * w[y].y;
                        p.v += v * weight;
                        p.c = p.c + outer(v, Vec2{static_cast<float>(x), static_cast<float>(y)} - f) *
                                        (4 * n * weight);
                    }
                p.p += p.v * h;
                p.p.x = std::clamp(p.p.x, .035f, .965f);
                p.p.y = std::clamp(p.p.y, .035f, .965f);
                p.f = (Mat{} + p.c * h) * p.f;
                auto sv = decompose(p.f);
                Vec2 old = sv.s;
                // Educational yield envelopes; granular mode is a compression-only approximation,
                // not a calibrated Drucker-Prager constitutive law.
                float lo = mode == 0   ? .975f
                           : mode == 1 ? .98f
                                       : .94f,
                      hi = mode == 0   ? 1.0075f
                           : mode == 1 ? 1.f
                                       : 1.06f;
                sv.s.x = std::clamp(sv.s.x, lo, hi);
                sv.s.y = std::clamp(sv.s.y, lo, hi);
                p.plastic = std::clamp(p.plastic * old.x * old.y / (sv.s.x * sv.s.y), .6f, 1.4f);
                p.f = sv.u * Mat{sv.s.x, 0, 0, sv.s.y} * sv.v.transpose();
            }
        }
    }
    void render(Canvas &c) const override {
        c.clear();
        c.dot({.5f, .32f}, .105f, {.12f, .18f, .25f});
        c.line({.03f, .035f}, {.97f, .035f}, {.3f, .4f, .5f});
        for (const auto &p : particles) {
            Color base = mode == 0   ? Color{.75f, .88f, 1}
                         : mode == 1 ? Color{.95f, .62f, .2f}
                                     : Color{.85f, .3f, .35f};
            c.dot(p.p, .002f, mix(base, {.2f, .8f, .8f}, p.v.length() * .3f));
        }
    }
    std::wstring status() const override {
        return std::format(L"{} particles | 64 x 64 grid | MLS-MPM / APIC", particles.size());
    }
    bool healthy() const override {
        for (const auto &p : particles)
            if (!finite(p.p.x) || !finite(p.p.y) || !finite(p.f.det()) || p.p.x < 0 || p.p.y < 0 ||
                p.p.x > 1 || p.p.y > 1)
                return false;
        return true;
    }
};
} // namespace
std::unique_ptr<Lab> makeMaterial() {
    return std::make_unique<Material>();
}
} // namespace physics
