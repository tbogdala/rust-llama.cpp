use std::io::{self, Write};

use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

mod common;

// an integration test using more settings to generate text
#[test]
pub fn predict_options_test() {
    let model_params = ModelOptions {
        n_gpu_layers: common::get_test_n_gpu_layers(),
        seed: -1,
        n_batch: 512,
        context_size: common::get_test_context_length(),
        ..Default::default()
    };

    let llm_model = match LLama::new(common::get_test_model_path(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let predict_options = PredictOptions {
        tokens: 512,
        threads: 16,
        batch: 512,
        temperature: 1.3,
        min_p: 0.05,
        penalty: 1.03,
        ignore_eos: true,
        stop_prompts: vec!["As an AI assisant:".to_string(), "OpenAI".to_string()],
        token_callback: Some(|token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        }),
        ..Default::default()
    };

    let prompt = "USER: Write the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.\nASSISTANT:";

    let result = llm_model.predict(prompt.to_string(), &predict_options);
    let (_, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64
    );
}
