use llama_cpp_rs::{options::{ModelOptions, PredictOptions}, LLama};

mod common;

#[test]
pub fn embeddings_test() {
    // when generating embeddings, the size of the embeddings is pulled from the model from
    // within the LLama wrapper.
    let model_params = ModelOptions {
        n_gpu_layers: common::get_test_n_gpu_layers(),
        context_size: common::get_test_context_length(),
        embeddings: true,
        ..Default::default()
    };

    let llm_model = match LLama::new(common::get_test_model_path(), &model_params) {
        Ok(m) => m,
        Err(err) => panic!("Failed to load model: {err}"),
    };

    let mut predict_options = PredictOptions {
        ..Default::default()
    };

    let test_prompt = "That is a happy person";
    let embeddings = llm_model.embeddings(test_prompt.to_string(), &mut predict_options).unwrap();

    // make sure we got some embeddings and that there's at least one that's not zero-ish.
    assert!(embeddings.len() > 0);
    assert!(embeddings.iter().any(|f| !f.eq(&(0.0 as f32))));

    // just make sure our test function works by comparing it to itself
    let sanity = cosine_similarity(&embeddings, &embeddings);
    assert!(sanity.eq(&1.0));

    let test_prompts = [
        "That is a very happy person",
        "That is a happy dog",
        "Behold an individual brimming with boundless joy and contentment",
        "Once upon a time in hollywood",
        "That's one small step for man, one giant leap for mankind.",
    ];

    println!("Comparing this prompt: \"{}\"", test_prompt);
    for test_prompt in test_prompts {
        let cmp_embeddings = llm_model.embeddings(test_prompt.to_string(), &mut predict_options).unwrap();
        let score = cosine_similarity(&embeddings, &cmp_embeddings);
        println!("  Similarity {}: \"{}\"", score, test_prompt);
    }
}

// Function to calculate the dot product of two vectors
fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(&a, &b)| a * b).sum()
}

// Function to calculate the magnitude of a vector
fn magnitude(vec: &[f32]) -> f32 {
    vec.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

// Function to calculate cosine similarity
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot = dot_product(vec1, vec2);
    let mag1 = magnitude(vec1);
    let mag2 = magnitude(vec2);

    if mag1 == 0.0 || mag2 == 0.0 {
        return 0.0; // To handle division by zero
    }

    dot / (mag1 * mag2)
}