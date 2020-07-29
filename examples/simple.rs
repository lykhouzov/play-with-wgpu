use futures::executor::block_on;
use play_with_webgpu::prelude::*;
use std::time::Instant;
use std::{borrow::Cow::Borrowed, convert::TryInto, mem::size_of};
use wgpu::*;
use zerocopy::AsBytes;
const RUN_TIMES: usize = 100;
const SIZE: usize = 512;
pub fn main() {
    // PREPARE DATA
    let gflop = (SIZE * SIZE * SIZE * 2) as f32 / 1e9;
    let mat_a = get_mat();
    let mat_b = get_mat();
    println!("mat_a = {:?}", &mat_a[0..15]);
    println!("mat_b = {:?}", &mat_b[0..15]);
    let (m, n, k): (u32, u32, u32) = (SIZE as u32, SIZE as u32, SIZE as u32);
    let result_matrix_size = ((m * n) as u64) * size_of::<f32>() as u64;
    //
    let mut result = Vec::new();
    let now = Instant::now();
    for _ in 0..RUN_TIMES {
        let (device, queue): (Device, Queue) = block_on(get_adapter());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(Borrowed("Command Encoder")),
        });
        let usage_flags = BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC;
        // Create a buffer for Matrix A
        let a_buffer = device.create_buffer_with_data(mat_a.as_slice().as_bytes(), usage_flags);
        // Create a buffer for Matrix B
        let b_buffer = device.create_buffer_with_data(mat_b.as_slice().as_bytes(), usage_flags);
        // Create a buffer for Result Matrix C
        let c_buffer = device.create_buffer(&BufferDescriptor {
            label: Some(Borrowed("matrix C")),
            size: result_matrix_size,
            usage: usage_flags,
            mapped_at_creation: false,
        });
        // Create MNK buffer
        let mnk_buffer = device.create_buffer_with_data(
            vec![m, n, k].as_slice().as_bytes(),
            usage_flags | BufferUsage::UNIFORM,
        );
        // CREATE GROUP LAYOUT
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: Borrowed(&[
                // matrix A
                BindGroupLayoutEntry::new(
                    0,
                    ShaderStage::COMPUTE,
                    BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: BufferSize::new(4),
                    },
                ),
                // matrix B
                BindGroupLayoutEntry::new(
                    1,
                    ShaderStage::COMPUTE,
                    BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: BufferSize::new(4),
                    },
                ),
                // matrix C - result matrix
                BindGroupLayoutEntry::new(
                    2,
                    ShaderStage::COMPUTE,
                    BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: BufferSize::new(4),
                    },
                ),
                // M, N K uniform
                BindGroupLayoutEntry::new(
                    3,
                    ShaderStage::COMPUTE,
                    BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: BufferSize::new(4),
                    },
                ),
            ]),
            label: Some(Borrowed("Group Layout")),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: Borrowed(&[&bind_group_layout]),
            push_constant_ranges: Borrowed(&[]),
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: ProgrammableStageDescriptor {
                module: &device.create_shader_module(include_spirv!("../shaders/sgemm1.spv")),
                entry_point: Borrowed("main"),
            },
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: Borrowed(&vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(a_buffer.slice(..)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(b_buffer.slice(..)),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(c_buffer.slice(..)),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(mnk_buffer.slice(..)),
                },
            ]),
            label: Some(Borrowed("Binding Group")),
        });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(m / 16, n / 16, 1);
        }
        result = block_on(read_buffer(
            (device, queue),
            encoder,
            &c_buffer,
            result_matrix_size,
        ));
    }
    let end = now.elapsed().as_secs_f32() / RUN_TIMES as f32;
    println!("GPU GLOPS = {:?}", gflop / end);
    println!("result = {:?}", &result[0..15]);
    println!("result.len = {:?}", &result.len());
}
async fn read_buffer(
    (device, queue): (Device, Queue),
    mut encoder: CommandEncoder,
    c_buffer: &Buffer,
    size: u64,
) -> Vec<f32> {
    let gpu_read_buffer = device.create_buffer(&BufferDescriptor {
        label: Some(Borrowed("GPU read buffer")),
        size,
        usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(c_buffer, 0, &gpu_read_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = gpu_read_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        let result = data
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        gpu_read_buffer.unmap();

        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}
fn get_mat() -> Vec<f32> {
    (1..=SIZE).map(|x| (x as f32 / SIZE as f32)).collect()
}
