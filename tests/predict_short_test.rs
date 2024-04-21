use std::{
    io::{self, Write},
    sync::Arc,
};

use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

mod common;

// This test is designed to make sure to cut a response shorter than hitting an EOS token. In
// the past, this has created memory management crashes because the output buffer was allocated incorrectly
#[test]
pub fn predict_short_tokens_test() {
    let model_params = ModelOptions {
        context_size: common::get_test_context_length(),
        n_gpu_layers: common::get_test_n_gpu_layers(),
        ..Default::default()
    };

    let mut llm_model = match LLama::new(common::get_test_model_path(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let predict_options = PredictOptions {
        tokens: 64,
        token_callback: Some(Arc::new(|token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        })),
        ..Default::default()
    };

    let prompt = "USER: What are the high level steps are required to implement a raytracing engine?\nASSISTANT:";

    let result = llm_model.predict(prompt.to_string(), &predict_options);
    let (_, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64
    );
}
