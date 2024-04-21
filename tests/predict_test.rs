use std::{
    io::{self, Write},
    sync::Arc,
};

use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

mod common;

// This test gives a large token request to generate an answer but should generally finish
// the response before that is filled.
#[test]
pub fn predict_test() {
    let model_params = ModelOptions {
        context_size: common::get_test_context_length(),
        n_gpu_layers: common::get_test_n_gpu_layers(),
        ..Default::default()
    };

    let mut llm_model = match LLama::new(common::get_test_model_path(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    // get slightly over half of the full context of the model in tokens. this way, when the second
    // prediciton request occurrs it will overflow the kv cache if things were not properly maintained.
    let mut predict_options = PredictOptions {
        tokens: (common::get_test_context_length() as f32 * 0.55) as i32,
        ignore_eos: true,
        token_callback: Some(Arc::new(move |token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        })),
        ..Default::default()
    };

    let prompt = "USER: What are the high level steps are required to implement a raytracing engine?\nASSISTANT:";

    let result = llm_model.predict(prompt.to_string(), &predict_options);
    let (prediction, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64
    );

    println!("Attempting second prediction...");

    // simulate a continue
    predict_options.tokens = 200;
    let continue_prompt = format!("{} {} ... Furthermore,", prompt, prediction);
    let result = llm_model.predict(continue_prompt, &predict_options);
    let (_, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64
    );
}
