// The following structures have documentation comments pulled from the llamacpp source headers where possible.

// Options controlling how the LLM model loads and behaves at runtime
#[derive(Debug, Clone)]
pub struct ModelOptions {
    // The context size of the model.
    pub context_size: i32,

    // RNG seed, -1 for random
    pub seed: i32,

    // prompt processing maximum batch size
    pub n_batch: i32,

    // force system to keep model in RAM
    pub m_lock: bool,

    // use mmap for faster loads, if possible
    pub m_map: bool,

    // only load the vocabulary, no weights
    pub vocab_only: bool,

    // embedding mode only
    pub embeddings: bool,

    // number of layers to store in VRAM
    pub n_gpu_layers: i32,

    // the GPU that is used for scratch and small tensors
    pub main_gpu: String,

    // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
    pub tensor_split: String,

    // attempt optimizations that help on some NUMA systems
    pub numa: bool,

    // RoPE base frequency, 0 = from model
    pub rope_freq_base: f32, 

    // RoPE frequency scaling factor, 0 = from model
    pub rope_freq_scale: f32,     
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            context_size: 512,
            seed: 0,
            m_lock: false,
            embeddings: false,
            vocab_only: false,
            m_map: true,
            n_batch: 0,
            numa: false,
            n_gpu_layers: 0,
            main_gpu: String::from(""),
            tensor_split: String::from(""),
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
        }
    }
}

// Options controlling the behavior of LLama functionality and text prediction.
pub struct PredictOptions {
    // RNG seed, -1 for random
    pub seed: i32,

    // number of threads to use for generation and batch processing
    // (llama.cpp will throttle this to 1 in cublas or vulkan are used with all layers offloaded to gpu)
    pub threads: i32,

    // the number of new tokens to predict
    pub tokens: i32,

    // sampler option: number of top probabilities to include.
    // <= 0 to use vocab size
    pub top_k: i32,

    // sampler option: last n tokens to penalize.
    // (0 = disable penalty, -1 = context size)
    pub repeat: i32,

    // prompt processing maximum batch size
    pub batch: i32,

    // number of tokens to keep from initial prompt when resetting context
    pub n_keep: i32,

    // sampler option: only the tokens with probabilities greater than or equal to the threshold will be included.
    // 1.0 = disabled
    pub top_p: f32,

    // sampler option: vase minimum probability threshold for token sampling (scaled by top token). for example,
    // a min_p setting of 0.05, with a top token at 0.90 probability, results in the minimum required probability 
    // becoming 4.5% (0.05 * 0.9).
    // 0.0 = disabled
    pub min_p: f32,

    // sampler option: scales the token probabilities. Values > 1.0 make less probable tokens more likely to be sampled.
    // Values < 1.0 makes it more likely to pick the most probable word according to its training data.
    // 1.0 = disabled
    pub temperature: f32,

    // sampler option: reduces the probability of generating tokens that have recently appeared in the generated text
    // 1.0 = disabled
    pub penalty: f32,

    // prints more verbose output when calling LLama::predict()
    pub debug_mode: bool,

    // strings to stop the prediction of text
    pub stop_prompts: Vec<String>,

    // ignore generated EOS tokens
    pub ignore_eos: bool,

    // sampler option: prioritizes sampling points identified by curvature changes in the probability distribution, 
    // aiming for diverse outputs beyond frequent & rare categories. Values > 1.0 might reinforce common outputs,
    // where values < 1.0 might encourages diversity and exploration.
    // 1.0 = disabled
    pub tail_free_sampling_z: f32,

    // sampler option: samples from "typical" regions of the probability space, balancing diversity & plausibility 
    // for more creative yet relevant outputs. Values > 1.0 focuses on regions with higher confidence resulting
    // in more plausible outputs but potentially less diverse. Values < 1.0 considers regions with broader range of probabilities, 
    // may lead to more diverse outputs but might include less common or expected results.
    // 1.0 = disabled
    pub typical_p: f32,

    // sampler option: repeat alpha frequency penalty
    // 0.0 = disabled
    pub frequency_penalty: f32,

    // sampler option: repeat alpha presence penalty
    // 0.0 = disabled
    pub presence_penalty: f32,

    // sampler option: an integer value indicating what version of mirostat to use.
    // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    pub mirostat: i32,

    // sampler option: target entropy for mirostat.
    pub mirostat_eta: f32,

    // sampler option: learning rate for mirostat
    pub mirostat_tau: f32,

    // if true, consider newlines as a repeatable token to be penalized
    pub penalize_nl: bool,

    // logit bias for specific tokens
    pub logit_bias: String,

    // optional callback function that receives tokens as they're predicted
    pub token_callback: Option<Box<dyn Fn(String) -> bool + Send + 'static>>,
    
    // path to file for saving/loading prompt eval state
    pub path_prompt_cache: String,

    // use mlock to keep model in memory
    pub m_lock: bool,

    // use mmap for faster loads
    pub m_map: bool,

    // save user input and generations to prompt cache
    pub prompt_cache_all: bool,

    // open the prompt cache read-only and do not update it
    pub prompt_cache_ro: bool,

    // the GPU that is used for scratch and small tensors
    pub main_gpu: String,

    // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
    pub tensor_split: String,

    // RoPE base frequency, 0 = from model
    pub rope_freq_base: f32, 

    // RoPE frequency scaling factor, 0 = from model
    pub rope_freq_scale: f32, 

    // number of tokens to draft during speculative decoding
    pub n_draft: i32,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            seed: -1,
            threads: 8,
            tokens: 128,
            top_k: 40,
            min_p: 0.0,
            repeat: 64,
            batch: 512,
            n_keep: 64,
            top_p: 0.95,
            temperature: 0.8,
            penalty: 1.1,
            debug_mode: false,
            stop_prompts: vec![],
            ignore_eos: false,
            tail_free_sampling_z: 1.0,
            typical_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: 0,
            mirostat_eta: 0.1,
            mirostat_tau: 5.0,
            penalize_nl: false,
            logit_bias: String::from(""),
            token_callback: None,
            path_prompt_cache: String::from(""),
            m_lock: false,
            m_map: true,
            prompt_cache_all: false,
            prompt_cache_ro: false,
            main_gpu: String::from(""),
            tensor_split: String::from(""),
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            n_draft: 16,
        }
    }
}
