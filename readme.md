This repository was ispired by awesome [Tutorial: OpenCL SGEMM tuning for Kepler](https://cnugteren.github.io/tutorial/pages/page1.html).
## Goals
1. Implement matrix multiplication using [wgpu-rs](https://github.com/gfx-rs/wgpu-rs)
2. Build a simple nerual network using [wgpu-rs](https://github.com/gfx-rs/wgpu-rs)

## Code examples
- `examples/simple.rs` contains spagetti-code of `matmul` with performance mesuarement. 
- `examples/linear.rs` contains a usage of "Tensor" multiplication, where `Tensor` is just a container for `data: Vec<f32>` and `dim: [usize;2]` which helps to build a wgpu `Buffer`