use std::{
    io::{self, Write},
    path::Path,
    sync::Arc,
};

use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

use serde_json::Value;

mod common;

// an integration test using more settings to generate text
#[test]
pub fn predict_grammar_test() {
    let model_params = ModelOptions {
        n_gpu_layers: common::get_test_n_gpu_layers(),
        seed: -1,
        n_batch: 512,
        context_size: common::get_test_context_length(),
        ..Default::default()
    };

    // the bundled json.gbnf tends to allow for newline spam, so we're using json_arr.gbnf for the example.
    let grammar_filepath = Path::new("llama.cpp/grammars/json_arr.gbnf");
    let grammar_string = std::fs::read_to_string(grammar_filepath).unwrap();

    let mut llm_model = match LLama::new(common::get_test_model_path(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let predict_options = PredictOptions {
        tokens: 512,
        batch: 512,
        temperature: 1.35,
        top_k: 33,
        top_p: 0.64,
        penalty: 1.13,
        grammar: grammar_string,
        token_callback: Some(Arc::new(|token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            true
        })),
        ..Default::default()
    };

    let input_sequence = "USER: ";
    let output_sequence = "ASSISTANT: ";
    let prompt = format!("{}You are an award-winning fiction author, known for jaded and sarcastic commentary on the human condition, contracted to write data files that describe engaging characters for video games. Create a new and *totally unique* character described in the JSON text format with fields for `description`, `personality` and `attributes`. Create a verbose character description, adding your trademark flair for engaging character design! It is *VITAL* that you follow all these instructions because this video game is very important to my career and I'll be fired from my job if it isn't good.\n{}", input_sequence, output_sequence);

    let result = llm_model.predict(prompt.to_string(), &predict_options);
    let (character_json_str, timings) = result.unwrap();
    println!(
        "\n\nTiming Data: {} tokens total in {:.2} ms ; {:.2} T/s\n",
        timings.n_eval,
        (timings.t_end_ms - timings.t_start_ms),
        1e3 / (timings.t_end_ms - timings.t_start_ms) * timings.n_eval as f64
    );

    // attempt to deserialize it, which *should* always work, thanks to the grammar
    let _parsed_value: Value = serde_json::from_str(character_json_str.trim()).unwrap();
}
