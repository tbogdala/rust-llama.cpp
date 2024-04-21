use std::{
    collections::HashMap,
    error::Error,
    ffi::{c_char, c_void, CStr, CString},
    mem::size_of,
    sync::Mutex,
};

use options::{ModelOptions, PredictOptions, TokenCallback};

use lazy_static::lazy_static;

pub mod options;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

lazy_static! {
    static ref CALLBACKS: Mutex<HashMap<usize, TokenCallback>> = Mutex::new(HashMap::new());
}

#[derive(Debug, Clone)]
pub struct LLama {
    ctx: *mut c_void,
    model: *mut c_void,
    prompt_cache: *mut c_void,
    embeddings: bool,
    context_size: i32,
}

impl LLama {
    // Creates a new LLama model context, state hidden internally, using the ModelOptions provided.
    pub fn new(model: String, opts: &ModelOptions) -> Result<Self, Box<dyn Error>> {
        let model_path = CString::new(model).unwrap();

        let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

        let main_gpu = main_gpu_cstr.as_ptr();

        let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

        let tensor_split = tensor_split_cstr.as_ptr();

        unsafe {
            let result = load_model(
                model_path.as_ptr(),
                opts.context_size,
                opts.seed,
                opts.m_lock,
                opts.embeddings,
                opts.m_map,
                opts.vocab_only,
                opts.n_gpu_layers,
                opts.n_batch,
                main_gpu,
                tensor_split,
                opts.rope_freq_base,
                opts.rope_freq_scale,
            );

            if result.ctx == std::ptr::null_mut() || result.model == std::ptr::null_mut() {
                return Err("Failed to load model".into());
            } else {
                Ok(Self {
                    ctx: result.ctx,
                    model: result.model,
                    prompt_cache: std::ptr::null_mut(),
                    embeddings: opts.embeddings,
                    context_size: opts.context_size,
                })
            }
        }
    }

    // Frees the encapsulated state object being wrapped by this class.
    pub fn free_model(&mut self) {
        if !self.prompt_cache.is_null() {
            unsafe {
                llama_free_prompt_cache(self.prompt_cache);
            }
            self.prompt_cache = std::ptr::null_mut();
        }
        if !self.ctx.is_null() {
            clear_callback(self.ctx);
            unsafe {
                llama_binding_free_model(self.ctx, self.model);
            }
            self.ctx = std::ptr::null_mut();
            self.model = std::ptr::null_mut();
        }
    }

    pub fn load_state(&self, state: String) -> Result<(), Box<dyn Error>> {
        let d = CString::new(state).unwrap().into_raw();
        let w = CString::new("rb").unwrap().into_raw();

        unsafe {
            let result = load_state(self.ctx, d, w);

            if result != 0 {
                return Err("Failed to load state".into());
            } else {
                Ok(())
            }
        }
    }

    pub fn save_state(&self, dst: String) -> Result<(), Box<dyn Error>> {
        let d = CString::new(dst.clone()).unwrap().into_raw();
        let w = CString::new("wb").unwrap().into_raw();

        unsafe {
            save_state(self.ctx, d, w);
        };

        std::fs::metadata(dst).map_err(|_| "Failed to save state".to_string())?;

        Ok(())
    }

    pub fn token_embeddings(
        &self,
        tokens: Vec<i32>,
        opts: &mut PredictOptions,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if !self.embeddings {
            return Err("model loaded without embeddings".into());
        }

        if opts.tokens == 0 {
            opts.tokens = 99999999;
        }

        unsafe {
            let embedding_size: i32 = get_llama_n_embd(self.model);

            let mut out = Vec::with_capacity((embedding_size) as usize);
            let mut my_array: Vec<i32> =
                Vec::with_capacity(opts.tokens as usize * size_of::<i32>());

            for (i, &v) in tokens.iter().enumerate() {
                my_array[i] = v;
            }

            let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

            let logit_bias = logit_bias_cstr.as_ptr();

            let grammar_cstr = CString::new(opts.grammar.clone()).unwrap();

            let grammar = grammar_cstr.as_ptr();

            let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

            let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

            let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

            let main_gpu = main_gpu_cstr.as_ptr();

            let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

            let tensor_split = tensor_split_cstr.as_ptr();

            let input = CString::new("").unwrap();

            let params = llama_allocate_params(
                input.as_ptr(),
                opts.seed,
                opts.threads,
                opts.tokens,
                opts.top_k,
                opts.top_p,
                opts.min_p,
                opts.temperature,
                opts.penalty,
                opts.repeat,
                opts.ignore_eos,
                opts.batch,
                opts.n_keep,
                std::ptr::null_mut(),
                0,
                opts.tail_free_sampling_z,
                opts.typical_p,
                opts.frequency_penalty,
                opts.presence_penalty,
                opts.mirostat,
                opts.mirostat_eta,
                opts.mirostat_tau,
                opts.penalize_nl,
                logit_bias,
                path_prompt_cache,
                opts.prompt_cache_in_memory,
                opts.m_lock,
                opts.m_map,
                main_gpu,
                tensor_split,
                opts.file_prompt_cache_ro,
                opts.rope_freq_base,
                opts.rope_freq_scale,
                opts.n_draft,
                grammar,
            );

            let mut emb_count: i32 = 0;
            let ret = get_token_embeddings(
                params,
                self.ctx,
                my_array.as_mut_ptr(),
                my_array.len() as i32,
                out.as_mut_ptr(),
                &mut emb_count,
            );
            llama_free_params(params);

            if ret != 0 {
                return Err("Embedding inference failed".into());
            }

            out.set_len(emb_count as usize);
            Ok(out)
        }
    }

    // Generates a f32 vector for the given `text` using the loaded model to generate embeddings.
    pub fn embeddings(
        &self,
        text: String,
        opts: &mut PredictOptions,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if !self.embeddings {
            return Err("model loaded without embeddings".into());
        }

        let c_str = CString::new(text.clone()).unwrap();

        let input = c_str.as_ptr();

        if opts.tokens == 0 {
            opts.tokens = 99999999;
        }

        let reverse_count = opts.stop_prompts.len();

        let mut c_strings: Vec<CString> = Vec::new();

        let mut reverse_prompt = Vec::with_capacity(reverse_count);

        let mut pass: *mut *const c_char = std::ptr::null_mut();

        for prompt in &opts.stop_prompts {
            let c_string = CString::new(prompt.clone()).unwrap();
            reverse_prompt.push(c_string.as_ptr());
            c_strings.push(c_string);
        }

        if !reverse_prompt.is_empty() {
            pass = reverse_prompt.as_mut_ptr();
        }

        unsafe {
            let embedding_size: i32 = get_llama_n_embd(self.model);

            let mut out = Vec::with_capacity((embedding_size) as usize);

            let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

            let logit_bias = logit_bias_cstr.as_ptr();

            let grammar_cstr = CString::new(opts.grammar.clone()).unwrap();

            let grammar = grammar_cstr.as_ptr();

            let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

            let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

            let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

            let main_gpu = main_gpu_cstr.as_ptr();

            let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

            let tensor_split = tensor_split_cstr.as_ptr();

            let params = llama_allocate_params(
                input,
                opts.seed,
                opts.threads,
                opts.tokens,
                opts.top_k,
                opts.top_p,
                opts.min_p,
                opts.temperature,
                opts.penalty,
                opts.repeat,
                opts.ignore_eos,
                opts.batch,
                opts.n_keep,
                pass,
                reverse_count as i32,
                opts.tail_free_sampling_z,
                opts.typical_p,
                opts.frequency_penalty,
                opts.presence_penalty,
                opts.mirostat,
                opts.mirostat_eta,
                opts.mirostat_tau,
                opts.penalize_nl,
                logit_bias,
                path_prompt_cache,
                opts.prompt_cache_in_memory,
                opts.m_lock,
                opts.m_map,
                main_gpu,
                tensor_split,
                opts.file_prompt_cache_ro,
                opts.rope_freq_base,
                opts.rope_freq_scale,
                opts.n_draft,
                grammar,
            );

            let mut emb_count: i32 = 0;
            let ret = get_embeddings(params, self.ctx, out.as_mut_ptr(), &mut emb_count);
            llama_free_params(params);

            if ret != 0 {
                return Err("Embedding inference failed".into());
            }

            out.set_len(emb_count as usize);
            Ok(out)
        }
    }

    // Does text inference with the loaded model given the `text` prompt and controlled by the PredictOptions
    // passed in. If `token_callback` is set in `opts`, that function will be called each time
    // a new token is predicted.
    // The function returns a tuple of the predicted string and the timing data for the prediction.
    pub fn predict(
        &mut self,
        text: String,
        include_specials: bool,
        opts: &PredictOptions,
    ) -> Result<(String, LLamaPredictTimings), Box<dyn Error>> {
        let c_str = CString::new(text.clone()).unwrap();

        let input = c_str.as_ptr();
        if let Some(callback) = opts.token_callback.as_ref() {
            set_callback(self.ctx, Some(callback.clone()));
        }

        let reverse_count = opts.stop_prompts.len();

        let mut c_strings: Vec<CString> = Vec::new();

        let mut reverse_prompt = Vec::with_capacity(reverse_count);

        let mut pass: *mut *const c_char = std::ptr::null_mut();

        for prompt in &opts.stop_prompts {
            let c_string = CString::new(prompt.clone()).unwrap();
            reverse_prompt.push(c_string.as_ptr());
            c_strings.push(c_string);
        }

        if !reverse_prompt.is_empty() {
            pass = reverse_prompt.as_mut_ptr();
        }

        // assume a little on the heavy side at 4 characters per token, then multiply by 4 again for
        // a character size of 4 bytes. also, we allocate a buffer to handle the whole context size.
        // FIXME: a better solution might be passing in a length as well to llama_predict and abort generation
        // when that limit is hit instead of allowing an overflow.
        assert!(self.context_size > 0);
        let mut out = Vec::with_capacity((self.context_size * 4 * 4) as usize);

        let logit_bias_cstr = CString::new(opts.logit_bias.clone()).unwrap();

        let logit_bias = logit_bias_cstr.as_ptr();

        let grammar_cstr = CString::new(opts.grammar.clone()).unwrap();

        let grammar = grammar_cstr.as_ptr();

        let path_prompt_cache_cstr = CString::new(opts.path_prompt_cache.clone()).unwrap();

        let path_prompt_cache = path_prompt_cache_cstr.as_ptr();

        let main_gpu_cstr = CString::new(opts.main_gpu.clone()).unwrap();

        let main_gpu = main_gpu_cstr.as_ptr();

        let tensor_split_cstr = CString::new(opts.tensor_split.clone()).unwrap();

        let tensor_split = tensor_split_cstr.as_ptr();

        unsafe {
            let params = llama_allocate_params(
                input,
                opts.seed,
                opts.threads,
                opts.tokens,
                opts.top_k,
                opts.top_p,
                opts.min_p,
                opts.temperature,
                opts.penalty,
                opts.repeat,
                opts.ignore_eos,
                opts.batch,
                opts.n_keep,
                pass,
                reverse_count as i32,
                opts.tail_free_sampling_z,
                opts.typical_p,
                opts.frequency_penalty,
                opts.presence_penalty,
                opts.mirostat,
                opts.mirostat_eta,
                opts.mirostat_tau,
                opts.penalize_nl,
                logit_bias,
                path_prompt_cache,
                opts.prompt_cache_in_memory,
                opts.m_lock,
                opts.m_map,
                main_gpu,
                tensor_split,
                opts.file_prompt_cache_ro,
                opts.rope_freq_base,
                opts.rope_freq_scale,
                opts.n_draft,
                grammar,
            );

            let ret = llama_predict(
                params,
                self.ctx,
                self.model,
                include_specials,
                out.as_mut_ptr(),
                self.prompt_cache,
            );
            llama_free_params(params);
            if ret.result != 0 {
                return Err("Failed to predict".into());
            }

            // upate the prompt cache opaque pointer
            self.prompt_cache = ret.prompt_cache;

            let c_str: &CStr = CStr::from_ptr(out.as_mut_ptr());
            let mut res: String = c_str.to_str().unwrap().to_owned();

            res = res.trim_start().to_string();
            res = res.trim_start_matches(&text).to_string();
            res = res.trim_start_matches('\n').to_string();

            for s in &opts.stop_prompts {
                res = res.trim_end_matches(s).to_string();
            }

            let timings = LLamaPredictTimings {
                t_start_ms: ret.t_start_ms,
                t_end_ms: ret.t_end_ms,
                t_load_ms: ret.t_load_ms,
                t_sample_ms: ret.t_sample_ms,
                t_p_eval_ms: ret.t_p_eval_ms,
                t_eval_ms: ret.t_eval_ms,
                n_sample: ret.n_sample,
                n_p_eval: ret.n_p_eval,
                n_eval: ret.n_eval,
            };

            Ok((res, timings))
        }
    }
}

pub struct LLamaPredictTimings {
    pub t_start_ms: f64,
    pub t_end_ms: f64,
    pub t_load_ms: f64,
    pub t_sample_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_sample: i32,
    pub n_p_eval: i32,
    pub n_eval: i32,
}

impl Drop for LLama {
    fn drop(&mut self) {
        self.free_model();
    }
}

fn set_callback(ctx: *mut c_void, callback: Option<TokenCallback>) {
    let mut callbacks = CALLBACKS.lock().unwrap();
    if let Some(callback) = callback {
        callbacks.insert(ctx as usize, callback);
    }
}

#[allow(dead_code)]
fn clear_callback(ctx: *mut c_void) {
    let mut callbacks = CALLBACKS.lock().unwrap();
    callbacks.remove(&(ctx as usize));
}

#[no_mangle]
extern "C" fn tokenCallback(ctx: *mut c_void, token: *const c_char) -> bool {
    let mut callbacks = CALLBACKS.lock().unwrap();

    if let Some(callback) = callbacks.get_mut(&(ctx as usize)) {
        let c_str: &CStr = unsafe { CStr::from_ptr(token) };
        let string = c_str.to_string_lossy().to_string();
        return callback(string);
    }

    true
}
