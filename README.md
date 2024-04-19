# Forked from mdrokz/rust-llama.cpp

Main changes from the forked version:

- [x] CHANGE: Updated llama.cpp submodule to tag [b2699](https://github.com/ggerganov/llama.cpp/tree/b2699).
- [x] ADDED:  Documentation for the structures and wrapper classes.
- [x] ADDED:  `LLama::predict()` integration tests.
- [x] FIXED:  Fixed a memory allocation error in `predict()` for the output buffer causing problems on free.
- [x] ADDED:  `get_llama_n_embd()` in the bindings to pull the embedding size generated directly from the model.
- [x] FIXED:  `LLama::embeddings()` so that it's functional and correctly obtains the floats for the embeddings.
              This includes a reworking of the C code to match the llamacpp embeddings sample.
- [x] ADDED:  `LLama::embeddings()` integration test with a sample cosine similarity test rig.
- [ ] FIXED?: `LLama::token_embeddings()` was given the same treatment as `LLama::embeddings()`, but is currently
              untested and no unit tests cover it.
- [x] DEL:    `ModelOptions::low_vram` was removed since it was unused and not present in llamacpp.
- [x] ADDED:  `ModelOptions::{rope_freq_base, rope_freq_scale, n_draft}` added
- [x] ADDED:  `PredictOptions::{rope_freq_base, rope_freq_scale}` added
- [x] DEL:    Removed all of the needless 'setters' for the options classes.
- [x] ADDED:  `PredictOptions::min_p` for min_p sampling.
- [x] CHANGE: `LLama::predict()` now returns a tuple in the result: the inferred text and a struct 
              containing misc data. It now requires `mut` access to self due to some data caching.
- [x] CHANGE: `load_model()` now returns a struct with both the loaded ctx and model. LLama now stores both pointers.
- [x] FIXED:  Fixed crashing from multiple `free_model()` invocations; updated basic_test integration test for verification.
- [x] FIXED:  Models now get their memory free'd now too instead of just the context.
- [x] FIXED:  Metal support on macos should work with the `metal` feature. Also added Accelerate support for macos
              if the `metal` feature is not enabled resulting in minor performance boosts. 
- [x] FIXED:  Cuda support on Win11 x64 should work with the `cuda` feature.
- [x] DEL:    Removed `load_model()` and `llama_allocate_params()` parameter `memory_f16` in the bindings 
              because it was removed upstream. Similarly, removed `ModelOptions::f16_memory` and
              `PredictOptions::f16_kv` to match.
- [x] CHANGE: The C++ bindings function `llama_predict()` got reworked to be in line with the
              current `llama.cpp/examples/main/main.cpp` example workflow.
- [x] FIXED:  `tokenCallback()` now uses `to_string_lossy()` in case the sent token is not valid UTF8.
- [x] ADDED:  Added a `logfile` feature to the crate. All logging statements in the bindings use the `LOG*`
              macros now and if that feature is enabled, a `llama.log` file will get created with the 
              output of all these `LOG*` macro calls.
- [x] ADDED:  The `lamma.cpp/llama.cpp` source file is now patched to use the `LOG` macro from their
              `lamma.cpp/common/log.h` file for logging. For now the GGML spam remains, but there should
              be no more output from this wrapper after that unless the `logfile` feature is enabled - 
              and even then, it should only get directed there.
- [x] CHANGE: Made `PredictOptions` clonable, swiched the container of the token callback to `Arc` and type aliased it.
              Renamed `prompt_cache_all` to `prompt_cache_in_memory` to highlight the fact that it controls the in-memory
              prompt caching mechanism while `path_prompt_cache` enables saving the data to file. Renamed `prompt_cache_ro`
              to `file_prompt_cache_ro` to highlight that it refers to leaving the file data read-only, and not the in-memory cache.
- [x] ADDED:  `LLama::predict()` now caches the state data for the processed prompt and attempts to reuse
              that state data to skip prompt processing if the _exact same_ prompt text is passed in. This feature
              is enabled by setting `PredictOptions::prompt_cache_in_memory` to true.
              NOTE: Please create an issue if you find a bug with its logic.
- [x] ADDED:  Support for [GBNF grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)
              through the `PredictOptions::grammar` string and added a new 'predict_grammar_test.rs' unit test
              as an example.
- [x] CHANGE: Removed numa options on model load for now; upstream implementation changed to support an enumeration
              of strategies, which can be implemented if there's a need for it.
- [x] DEL:    Removed `Llama::eval()` since it didn't return anything anyway and I'm not sure what the purpose of it ever was.
              The underlying function in llama.cpp has been deprecated for a while and was just officially removed.
              Even in skynet's Go wrappers this function is undocumented. If you need to test the evaluation of a prompt,
              just do a text prediction of length 1? If this was something you needed, open an issue and explain what
              it was being used for and I'll implement something.

This fork has the changes in development on the 'dev' branch, which will be merged into 'master'
once tested well enough.

Behavior of the original repo isn't guaranteed to stay the same! Any deviations should be mentioned
in the above list. **HOWEVER** ... if there's unit tests for a method, you can be sure some attention
has been paid to at least try to get it working in a reasonable manner. The version numbers for this fork
will also no longer keep parity with the original repo and will change as needed for this fork.


### Notes:

* Setting `ModelOptions::context_size` to zero will cause memory errors currently as that is what is used
  to create the buffers to send through the FFI.
* PLATFORMS: Windows 11, MacOS and Linux are tested without any features enabled. MacOS with the `metal`
  feature has been tested and should be functional. Windows 11 and Linux have been tested with `cuda` and
  should work.
* MAC: If some of the integration tests crash, consider the possibility that the `context_size` used in the test
  sends your mac out of memory. For example, my MBA M1 with 8gb of memory cannot handle a `context_size` of 
  4096 with 7B_Q4_K_S models and I have to drop the context down to 2048. Use the `RUST_LLAMA_CTX_LEN` env
  variable as described in the section below.
* WINDOWS: Make sure to install an llvm package to compile the bindings. I use scoop, so it's as 
  easy as running `scoop install llvm`. VS 2022 and Cuda 11.8 were also installed in addition to the rust
  toolchain (msvc version, the default) and the cargo commands were issued from the VS Developer Command Prompt.
  Note: The `logfile` feature is currently broken under Windows MSVC builds.
* If you're not using full GPU offloading, be mindful of the `threads` and `batch` values in your `PredictOptions`
  object that you're sending to `predict()`. CPU will work better with thread counts <= [number of actual CPU cores].
  For example, a Macbook Air M3 will choke on the `predict_options_test` (see below) with a thread count of `16` when
  the `metal` feature isn't enabled, but does just fine with `threads` set to `4`. On my machine you can even see it start
  to misstep with a value of `8`.

## Running tests

To run the tests, the library will need a GGUF model to load and use. The path
for this model is hardcoded to `models/model.gguf`. On a unix system you should
be able to create a symbolic link named `model.gguf` in that directory to the
GGUF model you wish to test with. FWIW, the test prompts use vicuna style prompts.

Any of the 'default' parameters for the integration tests should be modified
in the `tests/common/mod.rs` file.

The recommended way to run the tests involves using the correct feature for your
hardware accelleration. The following example is for CUDA device.

```bash
cargo test --release --features cuda --test "*" -- --nocapture --test-threads 1
```

With `--nocapture`, you'll be able to see the generated output. If it seems like
nothing is happening, make sure you're using the right feature for your system.
You also may wish to use the `--release` flag as well to speed up the tests.

Environment variables can be used to customize the test harness for a few parameters:

* `RUST_LLAMA_MODEL`: The relative filepath for the model to load (default is "models/model.gguf")
* `RUST_LLAMA_GPU_LAYERS`: The number of layers to offload to the gpu (default is 100)
* `RUST_LLAMA_CTX_LEN`: The context length to use for the test (default is 4096)

---

**Original README.md content below:**

---

# rust_llama.cpp
[![Docs](https://docs.rs/llama_cpp_rs/badge.svg)](https://docs.rs/llama_cpp_rs)
[![Crates.io](https://img.shields.io/crates/v/llama_cpp_rs.svg?maxAge=2592000)](https://crates.io/crates/llama_cpp_rs)

[LLama.cpp](https://github.com/ggerganov/llama.cpp) rust bindings.

The rust bindings are mostly based on https://github.com/go-skynet/go-llama.cpp/

## Building Locally

Note: This repository uses git submodules to keep track of [LLama.cpp](https://github.com/ggerganov/llama.cpp).

Clone the repository locally:

```bash
git clone --recurse-submodules https://github.com/mdrokz/rust-llama.cpp
```

```bash
cargo build
```

## Usage

```toml
[dependencies]
llama_cpp_rs = "0.3.0"
```

```rs
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn main() {
    let model_options = ModelOptions::default();

    let llama = LLama::new(
        "../wizard-vicuna-13B.ggmlv3.q4_0.bin".into(),
        &model_options,
    )
    .unwrap();

    let predict_options = PredictOptions {
        token_callback: Some(Box::new(|token| {
            println!("token1: {}", token);

            true
        })),
        ..Default::default()
    };

    llama
        .predict(
            "what are the national animals of india".into(),
             predict_options,
        )
        .unwrap();
}

```

## Examples 

The examples contain dockerfiles to run them

see [examples](https://github.com/mdrokz/rust-llama.cpp/blob/master/examples/README.md)

## TODO

- [x] Implement support for cublas,openBLAS & OpenCL [#7](https://github.com/mdrokz/rust-llama.cpp/pull/7)
- [x] Implement support for GPU (Metal)
- [ ] Add some test cases
- [ ] Support for fetching models through http & S3
- [x] Sync with latest master & support GGUF
- [x] Add some proper examples https://github.com/mdrokz/rust-llama.cpp/pull/7

## LICENSE

MIT
 
