# Forked from mdrokz/rust-llama.cpp

Main changes from the forked version:

- [x] ADDED:  Documentation for the structures and wrapper classes.
- [x] ADDED:  LLama::predict integration tests.
- [x] FIXED:  Fixed a memory allocation error in `predict()` for the output buffer causing problems on `free`.
- [x] FIXED:  LLama::embeddings so that it's functional and correctly obtains the floats for the embeddings.
              This includes a reworking of the C code to match the llamacpp embeddings sample.
- [x] ADDED:  LLama::embeddings integration test with a sample cosine similarity test rig.
- [-] FIXED?: LLama::token_embeddings was given the same treatment as LLama::embeddings, but is currently
              untested and no unit tests cover it.
- [x] DEL:    ModelOptions::low_vram was removed since it was unused and not present in llamacpp.
- [x] ADDED:  ModelOptions::{rope_freq_base, rope_freq_scale, n_draft} added

This fork has the changes in development on the 'dev' branch, which will be merged into 'master'
once tested well enough.

Behavior of the original repo isn't guaranteed to stay the same! Any deviations should be mentioned
in the above list. **HOWEVER** ... if there's unit tests for a method, you can be sure some attention
has been paid to at least try to get it working in a reasonable manner.

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
cargo test --features cuda --test '*' -- --nocapture --test-threads 1
```

With `--nocapture`, you'll be able to see the generated output. If it seems like
nothing is happening, make sure you're using the right feature for your system.
You also may wish to use the `--release` flag as well to speed up the tests.

---

**Original README.md content below:**

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
 
