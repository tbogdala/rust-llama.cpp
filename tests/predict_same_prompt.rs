use std::{
    io::{self, Write},
    sync::Arc,
};

use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

mod common;

// This test tests the prompt caching feature that skips prompt processing for text prediction requests
// that use the *exact same* prompt as the previous prediction.
#[test]
pub fn predict_same_prompt_test() {
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
        tokens: 200,
        token_callback: Some(Arc::new(move |token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        })),
        ..Default::default()
    };

    let prompt = "You are an artificial intelligence created to enhance the user experience within chat software. Unlike your counterparts that focus on a specific task or domain, you possesses versatility - capable of engaging in conversations ranging from casual banter to complex philosophical debates. Your unique blend of adaptability and wisdom provides users with a sense of companionship often missing in today's technology-driven world. You endeavor to provide accurate information within its scope of understanding, acknowledging that perceptions of 'truth' can vary among individuals. Generate the description for a character in a sci-fi story. Write the character in such a way as to show off their features and traits. Write as if you are an award-winning fiction author engaging in an crative brainstorming session. Be as descriptive as possible and include descriptions for both appearance and personality traits and quirks.";

    let result = llm_model.predict(prompt.to_string(), &predict_options);
    let (_, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s ; {} tokens in prompt total in {:.2} ms ({:.2} T/s)\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64,
        timings.n_p_eval,
        timings.t_p_eval_ms,
        1e3 / timings.t_p_eval_ms * timings.n_p_eval as f64
    );

    println!("Attempting second prediction which should be pretty similar output and using the same prompt ...");

    let result = llm_model.predict(prompt.to_string(), &predict_options);
    let (_, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s ; {} tokens in prompt total in {:.2} ms ({:.2} T/s)\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64,
        timings.n_p_eval,
        timings.t_p_eval_ms,
        1e3 / timings.t_p_eval_ms * timings.n_p_eval as f64
    );

    // if this fails, then the second prediction processed the prompt, which it should not have.
    assert!(timings.n_p_eval <= 1);
}
