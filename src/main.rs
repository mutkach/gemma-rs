use memmap2::Mmap;
use metal::{Buffer, PipelineBufferDescriptor};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs::File;

mod models;
use models::llama::llama::LlamaConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_file = File::open("data/llama-tiny/config.json")?;
    let config: LlamaConfig = serde_json::from_reader(config_file)?;

    println!("Model config: {}", config);

    let file = File::open("data/llama-tiny/llama-tiny.safetensors")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    println!("Loaded {} tensors", tensors.len());
    for (name, tensor) in &tensors.tensors() {
        println!("{}: shape={:?}", name, tensor.shape());
    }

    Ok(())
}
