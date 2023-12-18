use llama_cpp_rs::{options::ModelOptions, LLama};

mod common;

#[test]
pub fn load_test() {
    let model_params = ModelOptions {
        n_gpu_layers: common::N_GPU_LAYERS,
        ..Default::default()
    };

    let _llm_model = match LLama::new(common::MODEL_PATH.to_string(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };
}

