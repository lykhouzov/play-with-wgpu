#[allow(dead_code)]
pub mod tensor;
#[macro_use]
extern crate lazy_static;
use futures::executor::block_on;
use std::{
    borrow::{Cow, Cow::Borrowed},
    convert::TryInto,
    mem::size_of,
    path::PathBuf,
};
use wgpu::*;
use zerocopy::AsBytes;
pub mod prelude {
    pub use super::tensor::Tensor;
    pub use super::{get_adapter, matmul, matsum, set_template};
}

lazy_static! {
    static ref ADATPTER: (Device, Queue) = block_on(get_adapter());
    static ref SHADERS_PATH: PathBuf = {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("shaders");
        path
    };
}

static mut TEMPLATE: &'static str = "sgemm1";
pub fn set_template(tmpl: &'static str) {
    unsafe {
        TEMPLATE = tmpl;
    }
}
pub fn matmul<T>(a: &Vec<T>, b: &Vec<T>, m: usize, n: usize, k: usize) -> Vec<f32>
where
    T: AsBytes,
{
    let mut encoder = ADATPTER
        .0
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some(Borrowed("Command Encoder")),
        });
    let (m, n, k): (u32, u32, u32) = (m as u32, n as u32, k as u32);
    let result_matrix_size = ((m * n) as u64) * size_of::<T>() as u64;
    // CREATE BUFFERS
    let usage_flags = BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC;
    let a_buffer = get_buffer_from_vec(&mut encoder, &a, usage_flags, Some(Borrowed("matrix A")));
    let b_buffer = get_buffer_from_vec(&mut encoder, &b, usage_flags, Some(Borrowed("matrix B")));
    let c_buffer = ADATPTER.0.create_buffer(&BufferDescriptor {
        label: Some(Borrowed("matrix C")),
        size: result_matrix_size,
        usage: usage_flags,
        mapped_at_creation: false,
    });
    let mnk_buffer = get_buffer_from_vec(
        &mut encoder,
        &vec![m, n, k],
        BufferUsage::UNIFORM | BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        Some(Borrowed("MNK")),
    );
    let bind_group_layout = get_group_layout();
    let pipeline_layout = ADATPTER
        .0
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: Borrowed(&[&bind_group_layout]),
            push_constant_ranges: Borrowed(&[]),
        });
    let cs_module = ADATPTER
        .0
        .create_shader_module(include_spirv!("../shaders/sgemm1.spv"))
        // .create_shader_module(get_shader_code(unsafe { TEMPLATE }));
        ;
    let compute_pipeline = ADATPTER
        .0
        .create_compute_pipeline(&ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: ProgrammableStageDescriptor {
                module: &cs_module,
                entry_point: Borrowed("main"),
            },
        });

    let bindings: Vec<BindGroupEntry> = vec![
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
    ];
    let bind_group = get_bind_group(&bindings, &bind_group_layout);
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(m / 16, n / 16, 1)
    }
    block_on(read_buffer(encoder, &c_buffer, result_matrix_size))
}
async fn read_buffer(mut encoder: CommandEncoder, c_buffer: &Buffer, size: u64) -> Vec<f32> {
    let gpu_read_buffer = ADATPTER.0.create_buffer(&BufferDescriptor {
        label: Some(Borrowed("GPU read buffer")),
        size,
        usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&c_buffer, 0, &gpu_read_buffer, 0, size);
    ADATPTER.1.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = gpu_read_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    ADATPTER.0.poll(Maintain::Wait);

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
fn get_buffer_from_vec<'a, T>(
    _encoder: &mut CommandEncoder,
    v: &Vec<T>,
    usage: BufferUsage,
    _label: Option<Cow<'a, str>>,
) -> Buffer
where
    T: zerocopy::AsBytes,
{
    let v = v.as_slice().as_bytes();
    // let size = (v.len() * BYTE_SIZE) as u64;
    let buf = (&ADATPTER.0).create_buffer_with_data(v, usage);
    buf
}
fn get_group_layout() -> BindGroupLayout {
    ADATPTER
        .0
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        })
}

fn get_bind_group(bindings: &Vec<BindGroupEntry>, layout: &BindGroupLayout) -> BindGroup {
    ADATPTER.0.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: layout,
        entries: Borrowed(&bindings),
        label: Some(Borrowed("Binding Group")),
    })
}

pub async fn get_adapter() -> (Device, Queue) {
    let instance = wgpu::Instance::new(BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::Default,
            compatible_surface: None,
        })
        .await
        .unwrap();
    adapter
        .request_device(
            &DeviceDescriptor {
                features: Features::empty(),
                limits: Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap()
}
pub fn get_shader_code<'a>(name: &'static str) -> ShaderModuleSource<'a> {
    match name {
        _ => include_spirv!("../shaders/sgemm1.spv"),
    }
}

pub fn matsum(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(a, b)| *a + *b).collect()
}
