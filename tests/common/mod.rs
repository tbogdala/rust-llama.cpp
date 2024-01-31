// Put some common definitions in here for the tests so things can be changed in one spot
// or changed through environment variables

pub const MODEL_PATH: &str = "models/model.gguf";
pub const N_GPU_LAYERS: &str = "100";
pub const CONTEXT_LENGTH: &str = "4096";

// the relative path to the model to load for the tests
pub fn get_test_model_path() -> String {
    std::env::var("RUST_LLAMA_MODEL").unwrap_or(MODEL_PATH.to_string())
}

// the number of layers to offload to gpu; can be set to a large number like 1000
// to ensure all layers are offloaded.
pub fn get_test_n_gpu_layers() -> i32 {
    let var_str = std::env::var("RUST_LLAMA_GPU_LAYERS").unwrap_or(N_GPU_LAYERS.to_string());
    var_str.parse().unwrap_or(100)
}

// the size of the context to use for the model
#[allow(dead_code)]
pub fn get_test_context_length() -> i32 {
    let var_str = std::env::var("RUST_LLAMA_CTX_LEN").unwrap_or(CONTEXT_LENGTH.to_string());
    var_str.parse().unwrap_or(4096)
}
