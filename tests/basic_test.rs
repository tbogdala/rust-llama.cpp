use llama_cpp_rs::{options::ModelOptions, LLama};


#[test]
pub fn load_test() {
    let model_params = ModelOptions {
        n_gpu_layers: 100,
        ..Default::default()
    };

    let model_path = "models/model.gguf";
    let _llm_model = match LLama::new(model_path.to_string(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model from {model_path}: {err}"),
    };
}

