use llama_cpp_rs::{options::ModelOptions, LLama};

mod common;

#[test]
pub fn load_test() {
    let model_params = ModelOptions {
        n_gpu_layers: common::N_GPU_LAYERS,
        ..Default::default()
    };
    
    let mut llm_model = match LLama::new(common::MODEL_PATH.to_string(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };
    //std::thread::sleep(std::time::Duration::from_secs(5));
    println!("Model loading. Now attempting to free the model.");
    llm_model.free_model();

    println!("Model has been freed. A second free shouldn't crash.");
    llm_model.free_model();
}

