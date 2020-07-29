use play_with_webgpu::prelude::*;
use std::time::Instant;
const RUN_TIMES: usize = 100;
const SIZE: usize = 512;
fn main() {
    let gflop = (SIZE * SIZE * SIZE * 2) as f32 / 1e9;
    let input_1 = Tensor::randu(&[SIZE, SIZE]);
    let input_2 = Tensor::randu(&[SIZE, SIZE]);
    let now = Instant::now();
    (0..RUN_TIMES).collect::<Vec<_>>().iter().for_each(|_| {
        let _ = &input_1 * &input_2;
    });
    let end = now.elapsed().as_secs_f32() / RUN_TIMES as f32;
    println!("GPU GFLOPS = {:?}", gflop / end);
}
