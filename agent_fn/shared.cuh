FLAMEGPU_DEVICE_FUNCTION float vec2Length(int x, int y) { return sqrtf(((x * x) + (y * y))); }
FLAMEGPU_DEVICE_FUNCTION float vec2Dist(int x, int y, int xo, int yo) {
    return vec2Length(x - xo, y - yo);
}

constexpr int N_RESOURCE_TYPES = 2;
constexpr int N_DIM = 2;
constexpr int TARGET_RESOURCE_AMOUNT = 5;
