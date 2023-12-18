// Put some common definitions in here for the tests so things can be changed in one spot.

// the relative path to the model to load for the tests
pub const MODEL_PATH: &str = "models/model.gguf";

// the number of layers to offload to gpu; can be set to a large number like 1000
// to ensure all layers are offloaded.
pub const N_GPU_LAYERS: i32 = 100;

// this is the context size of the model. for llama 2 models, for example,
// it should be 4096 where llama 1 models should be 2048.
pub const NATIVE_CONTEXT_SIZE: i32 = 4096;