use std::path::PathBuf;
use std::{env, fs};

use cc::Build;

fn compile_bindings(out_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header("./binding.h")
        .blocklist_function("tokenCallback")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(&out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn compile_opencl(cx: &mut Build, cxx: &mut Build) {
    cx.flag("-DGGML_USE_CLBLAST");
    cxx.flag("-DGGML_USE_CLBLAST");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    }

    cxx.file("./llama.cpp/ggml-opencl.cpp");
}

fn compile_openblas(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/openblas")
        .include("/usr/local/include/openblas");
    println!("cargo:rustc-link-lib=openblas");
}

fn compile_blis(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/blis")
        .include("/usr/local/include/blis");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=blis");
}

fn compile_cuda(cxx_flags: &str, outdir: &PathBuf) {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");

    // if the environment variable isn't set, this will probably
    // impact the success chances of this build script.
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_default();
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
            cuda_path
        );
    }

    let libs = if cfg!(target_os = "linux") {
        "cuda culibos cublas cudart cublasLt pthread dl rt"
    } else {
        "cublas cudart cublasLt"
    };

    for lib in libs.split_whitespace() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut nvcc = cc::Build::new();

    let env_flags = vec![
        ("LLAMA_CUDA_DMMV_X=32", "-DGGML_CUDA_DMMV_X"),
        ("LLAMA_CUDA_DMMV_Y=1", "-DGGML_CUDA_DMMV_Y"),
        ("LLAMA_CUDA_KQUANTS_ITER=2", "-DK_QUANTS_PER_ITERATION"),
    ];

    let nvcc_flags = "--forward-unknown-to-host-compiler -arch=native ";

    for nvcc_flag in nvcc_flags.split_whitespace() {
        nvcc.flag(nvcc_flag);
    }

    for cxx_flag in cxx_flags.split_whitespace() {
        nvcc.flag(cxx_flag);
    }

    for env_flag in env_flags {
        let mut flag_split = env_flag.0.split("=");
        if let Ok(val) = std::env::var(flag_split.next().unwrap()) {
            nvcc.flag(&format!("{}={}", env_flag.1, val));
        } else {
            nvcc.flag(&format!("{}={}", env_flag.1, flag_split.next().unwrap()));
        }
    }

    // this next block is a hackjob but seems to work. any further cleanup
    // by people more specialized in windows development would be appreciated.

    if cfg!(target_os = "linux") {
        nvcc.compiler("nvcc")
            .file("./llama.cpp/ggml-cuda.cu")
            .flag("-Wno-pedantic")
            .include("./llama.cpp/ggml-cuda.h")
            .compile("ggml-cuda");
    } else {
        let include_path = format!("{}\\include", cuda_path);

        let object_file = outdir
            .join("llama.cpp")
            .join("ggml-cuda.o")
            .to_str()
            .expect("Could not build ggml-cuda.o filename")
            .to_string();

        std::process::Command::new("nvcc")
            .arg("-ccbin")
            .arg(
                cc::Build::new()
                    .get_compiler()
                    .path()
                    .parent()
                    .unwrap()
                    .join("cl.exe"),
            )
            .arg("-I")
            .arg(&include_path)
            .arg("-o")
            .arg(&object_file)
            .arg("-x")
            .arg("cu")
            .arg("-maxrregcount=0")
            .arg("--machine")
            .arg("64")
            .arg("--compile")
            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
            .arg("--generate-code=arch=compute_75,code=[compute_75,sm_75]")
            .arg("-D_WINDOWS")
            .arg("-DNDEBUG")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-D_CRT_SECURE_NO_WARNINGS")
            .arg("-D_MBCS")
            .arg("-DWIN32")
            .arg(r"-Illama.cpp\include\ggml")
            .arg(r"llama.cpp\ggml-cuda.cu")
            .status()
            .unwrap();

        nvcc.object(&object_file);
        nvcc.flag("-DGGML_USE_CUBLAS");
        nvcc.include(&include_path);
    }
}

fn compile_ggml(cx: &mut Build, cx_flags: &str) {
    for cx_flag in cx_flags.split_whitespace() {
        cx.flag(cx_flag);
    }

    cx.include("./llama.cpp")
        .file("./llama.cpp/ggml.c")
        .file("./llama.cpp/ggml-alloc.c")
        .file("./llama.cpp/ggml-backend.c")
        .file("./llama.cpp/ggml-quants.c")
        .cpp(false)
        .define("_GNU_SOURCE", None)
        .define("GGML_USE_K_QUANTS", None)
        .compile("ggml");
}

fn compile_metal(cx: &mut Build, cxx: &mut Build, out: &PathBuf) {
    cx.flag("-DGGML_USE_METAL").flag("-DGGML_METAL_NDEBUG");
    cxx.flag("-DGGML_USE_METAL");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalKit");

    // solution yoinked from rustformers/llm -- thanks @philpax
    // ref: https://github.com/rustformers/llm/commit/9d39ff8cc0a89bb22cc17bdc1dd2470f3421d788
    const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";
    println!("cargo:rerun-if-changed={GGML_METAL_METAL_PATH}");
    println!("cargo:rerun-if-changed={GGML_METAL_PATH}");

    // HACK: patch ggml-metal.m so that it includes ggml-metal.metal, so that
    // a runtime dependency is not necessary
    let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
        .expect("Could not read ggml-metal.metal")
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\"', "\\\"");

    let ggml_metal = std::fs::read_to_string(GGML_METAL_PATH).expect("Could not read ggml-metal.m");

    let needle = r#"NSString * src = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error];"#;
    if !ggml_metal.contains(needle) {
        panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated.");
    }

    // Replace the runtime read of the file with a compile-time string
    let ggml_metal = ggml_metal.replace(
        needle,
        &format!(r#"NSString * src  = @"{ggml_metal_metal}";"#),
    );

    // Replace the judicious use of `fprintf` with the already-existing `metal_printf`,
    // backing up the definition of `metal_printf` first
    let ggml_metal = ggml_metal
        .replace(
            r#"#define metal_printf(...) fprintf(stderr, __VA_ARGS__)"#,
            "METAL_PRINTF_DEFINITION",
        )
        .replace("fprintf(stderr,", "metal_printf(")
        .replace(
            "METAL_PRINTF_DEFINITION",
            r#"#define metal_printf(...) fprintf(stderr, __VA_ARGS__)"#,
        );

    let patched_ggml_metal_path = out.join("ggml-metal.m");
    std::fs::write(&patched_ggml_metal_path, ggml_metal)
        .expect("Could not write temporary patched ggml-metal.m");

    cx.file(patched_ggml_metal_path);
}

fn compile_llama(cxx: &mut Build, cxx_flags: &str, out_path: &PathBuf, ggml_type: &str) {
    for cxx_flag in cxx_flags.split_whitespace() {
        cxx.flag(cxx_flag);
    }

    let ggml_obj = PathBuf::from(&out_path).join("llama.cpp/ggml.o");

    cxx.object(ggml_obj);

    if !ggml_type.is_empty() {
        // the patched ggml-metal.o file has a prefix in the output directory so search it out
        if ggml_type.eq("metal") {
            for fs_entry in fs::read_dir(out_path).unwrap() {
                let fs_entry = fs_entry.unwrap();
                let path = fs_entry.path();
                if path.ends_with("-ggml-metal.o") {
                    cxx.object(path);
                    break;
                }
            }
        } else {
            let ggml_feature_obj =
                PathBuf::from(&out_path).join(format!("llama.cpp/ggml-{}.o", ggml_type));
            cxx.object(ggml_feature_obj);
        }
    }

    if cfg!(target_os = "windows") {
        // for some reason, this only appears to be needed under windows?
        let build_info_str = std::process::Command::new("sh")
            .arg("llama.cpp/scripts/build-info.sh")
            .output()
            .expect("Failed to generate llama.cpp/common/build-info.cpp from the shell script.");

        let mut build_info_file = fs::File::create("llama.cpp/common/build-info.cpp")
            .expect("Could not create llama.cpp/common/build-info.cpp file");
        std::io::Write::write_all(&mut build_info_file, &build_info_str.stdout)
            .expect("Could not write to llama.cpp/common/build-info.cpp file");

        cxx.shared_flag(true)
            .file("./llama.cpp/common/build-info.cpp");
    }

    // HACK: gonna use the same trick used to patch the metal shader into ggml
    // to redirect the logging macros in llama.cpp to the LOG macro in log.h.
    const LLAMACPP_PATH: &str = "llama.cpp/llama.cpp";
    const PATCHED_LLAMACPP_PATH: &str = "llama.cpp/llama-patched.cpp";
    let llamacpp_code =
        std::fs::read_to_string(LLAMACPP_PATH).expect("Could not read llama.cpp source file.");
    let needle1 =
        r#"#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)"#;
    let needle2 =
        r#"#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)"#;
    let needle3 =
        r#"#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)"#;
    if !llamacpp_code.contains(needle1)
        || !llamacpp_code.contains(needle2)
        || !llamacpp_code.contains(needle3)
    {
        panic!("llama.cpp does not contain the needles to be replaced; the patching logic needs to be reinvestigated!");
    }
    let patched_llamacpp_code = llamacpp_code
        .replace(
            needle1,
            "#include \"log.h\"\n#define LLAMA_LOG_INFO(...)  LOG(__VA_ARGS__)",
        )
        .replace(needle2, "#define LLAMA_LOG_WARN(...)  LOG(__VA_ARGS__)")
        .replace(needle3, "#define LLAMA_LOG_ERROR(...) LOG(__VA_ARGS__)");
    std::fs::write(&PATCHED_LLAMACPP_PATH, patched_llamacpp_code)
        .expect("Attempted to write the patched llama.cpp file out to llama-patched.cpp");

    cxx.shared_flag(true)
        .file("./llama.cpp/common/common.cpp")
        .file("./llama.cpp/common/sampling.cpp")
        .file("./llama.cpp/common/grammar-parser.cpp")
        .file("./llama.cpp/llama-patched.cpp")
        .file("./binding.cpp")
        .cpp(true)
        .compile("binding");
}

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));

    compile_bindings(&out_path);

    let mut cx_flags = String::from("");
    let mut cxx_flags = String::from("");

    // check if os is linux
    // if so, add -fPIC to cxx_flags
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        cx_flags.push_str(" -std=c11 -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -pthread -march=native -mtune=native");
        cxx_flags.push_str(" -std=c++11 -Wall -Wdeprecated-declarations -Wunused-but-set-variable -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -fPIC -pthread -march=native -mtune=native");
    } else if cfg!(target_os = "windows") {
        cx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
        cxx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
    }

    let mut cx = cc::Build::new();
    let mut cxx = cc::Build::new();
    let mut ggml_type = String::new();

    cxx.include("./llama.cpp/common")
        .include("./llama.cpp")
        .include("./include_shims");

    if cfg!(feature = "opencl") {
        compile_opencl(&mut cx, &mut cxx);
        ggml_type = "opencl".to_string();
    } else if cfg!(feature = "openblas") {
        compile_openblas(&mut cx);
    } else if cfg!(feature = "blis") {
        compile_blis(&mut cx);
    } else if cfg!(feature = "metal") && cfg!(target_os = "macos") {
        compile_metal(&mut cx, &mut cxx, &out_path);
        ggml_type = "metal".to_string();
    }

    if !cfg!(feature = "metal") && cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        cx.define("GGML_USE_ACCELERATE", None);
    }

    if cfg!(feature = "cuda") {
        cx_flags.push_str(" -DGGML_USE_CUBLAS");
        cxx_flags.push_str(" -DGGML_USE_CUBLAS");

        if cfg!(target_os = "linux") {
            cx.include("/usr/local/cuda/include")
                .include("/opt/cuda/include");
            cxx.include("/usr/local/cuda/include")
                .include("/opt/cuda/include");

            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                cx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
                cxx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
            }
        } else {
            cx.flag("/MT");
            cxx.flag("/MT");
        }

        compile_ggml(&mut cx, &cx_flags);

        compile_cuda(&cxx_flags, &out_path);

        if !cfg!(feature = "logfile") {
            cxx.define("LOG_DISABLE_LOGS", None);
        }
        compile_llama(&mut cxx, &cxx_flags, &out_path, "cuda");
    } else {
        compile_ggml(&mut cx, &cx_flags);

        if !cfg!(feature = "logfile") {
            cxx.define("LOG_DISABLE_LOGS", None);
        }
        compile_llama(&mut cxx, &cxx_flags, &out_path, &ggml_type);
    }
}
