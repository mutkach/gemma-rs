use metal::{CompileOptions, ComputePipelineState, Device};
use serde::{Deserialize, Serialize};

const LLAMA_KERNELS: &str = r#"
kernel void my_kernel(const device float* input [[buffer(0)],
                       device float* output [[buffer(1)],
                       uint id [[thread_position_in_grid]]) {
    // TODO: implement attention computation
    output[id] = input[id];
}
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub factor: f64,
    pub type_: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub hidden_act: String,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub model_type: String,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub num_key_value_heads: i64,
    pub pretraining_tp: i64,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: i64,
}

impl std::fmt::Display for LlamaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LlamaConfig {{ hidden_size: {}, num_hidden_layers: {}, vocab_size: {} }}",
            self.hidden_size, self.num_hidden_layers, self.vocab_size
        )
    }
}

struct LlamaBlock {
    input_tensor: metal::Buffer,
    attention_weights: Vec<metal::Buffer>,
    attention_kernel: metal::Function,
    attention_bias: Vec<metal::Buffer>,

    norm: metal::Function,
    norm_bias: Vec<metal::Buffer>,
    mlp_bias: Vec<metal::Buffer>,
    mlp_kernel: metal::Function,
}

//impl LlamaBlock {
//    fn new(device: &Device) -> Self {
//        let library = device.new_library_with_source(LLAMA_KERNELS, &CompileOptions::new())?;
//        let norm = library.get_function("rms_norm", None)?;
//        let attention = library.get_function("attention", None)?;
//        LlamaBlock {
//            input_tensor: device.new_buffer(0, metal::MTLResourceOptions::StorageModeShared),
//            attention_weights: vec![],
//            attention_kernel: attention,
//            attention_bias: vec![],
//            norm,
//            norm_bias: vec![],
//            mlp_bias: vec![],
//            mlp_kernel: attention,
//        }
//    }
//}

struct LlamaEmbedding {
    weight: metal::Buffer,
    bias: Option<metal::Buffer>,
}

struct Llama {
    blocks: Vec<LlamaBlock>,
    embedding: LlamaEmbedding,
    lm_head: LlamaEmbedding,
    output: metal::Buffer,
}

impl Llama {
    pub fn new(config: &LlamaConfig) -> Self {
        todo!()
    }
}
