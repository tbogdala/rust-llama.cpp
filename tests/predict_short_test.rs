use std::io::{self, Write};

use llama_cpp_rs::{options::{ModelOptions, PredictOptions}, LLama};

mod common;

// This test is designed to make sure to cut a response shorter than hitting an EOS token. In
// the past, this has created memory management crashes because the output buffer was allocated incorrectly
#[test]
pub fn predict_short_tokens_test() {
    let model_params = ModelOptions {
        n_gpu_layers: common::N_GPU_LAYERS,
        ..Default::default()
    };

    let llm_model = match LLama::new(common::MODEL_PATH.to_string(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let predict_options = PredictOptions {
        tokens: 64,
        token_callback: Some(Box::new(|token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        })),
        ..Default::default()
    };

    let prompt = "USER: What are the high level steps are required to implement a raytracing engine?\nASSISTANT: ";

    let _result = llm_model.predict(prompt.to_string(), predict_options);
}
