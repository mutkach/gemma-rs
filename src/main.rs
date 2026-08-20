use memmap2::Mmap;
use metal::{Buffer, Device, MTLResourceOptions, PipelineBufferDescriptor};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File};

mod models;
use models::llama::llama::LlamaConfig;

pub enum LayerType {
    Attention,
    FFN,
    Embedding,
    Output,
}

struct TensorMeta {
    dtype: Dtype,
    offset: usize,
    shape: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::system_default().expect("Metal Device not found");
    println!("Using Metal device: {}", device.name());

    let config_file = File::open("data/llama-tiny/config.json")?;
    let config: LlamaConfig = serde_json::from_reader(config_file)?;

    println!("Model config: {}", config);

    let file = File::open("data/llama-tiny/llama-tiny.safetensors")?;

    let mmap = unsafe { Mmap::map(&file)? };
    let base_ptr = mmap.as_ptr() as usize;
    let tensors = SafeTensors::deserialize(&mmap)?;

    let metal_buffer: Buffer = device.new_buffer_with_bytes_no_copy(
        mmap.as_ptr() as *mut _,
        mmap.len() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );

    // !TODO: sort weights before mmapping

    println!("Loaded {} tensors", tensors.len());

    let mut tensor_index: HashMap<String, TensorMeta> = HashMap::new();

    for (name, tensor) in tensors.tensors() {
        let tensor_data_ptr = tensor.data().as_ptr() as usize;
        let dtype = tensor.dtype();
        tensor_index.insert(
            name.clone(),
            TensorMeta {
                dtype: tensor.dtype(),
                offset: tensor_data_ptr - base_ptr,
                shape: tensor.shape().into(),
            },
        );

        println!("{}: shape={:?} of dtype={}", name, tensor.shape(), dtype);
    }

    Ok(())
}
