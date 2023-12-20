use std::io::{self, Write};

use llama_cpp_rs::{options::{ModelOptions, PredictOptions}, LLama};

mod common;

// This test gives a large token request to generate an answer but should generally finish
// the response before that is filled.
#[test]
pub fn predict_test() {
    let model_params = ModelOptions {
        n_gpu_layers: common::N_GPU_LAYERS,
        ..Default::default()
    };

    let llm_model = match LLama::new(common::MODEL_PATH.to_string(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let predict_options = PredictOptions {
        tokens: 4096,
        token_callback: Some(Box::new(|token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        })),
        ..Default::default()
    };

    let prompt = "USER: What are the high level steps are required to implement a raytracing engine?\nASSISTANT:";

    let result = llm_model.predict(prompt.to_string(), predict_options);
    if let Ok((_, timings)) = result {
        println!("\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n",
            timings.n_eval, 
            (timings.t_end_ms - timings.t_start_ms),
            1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64);
    }
}
