extern crate cc;
// extern crate bindgen;
// use std::path::PathBuf;
// use std::env;

// rust+cuda references
//  - https://github.com/MWATelescope/mwa_hyperdrive
//  - https://github.com/termoshtt/link_cuda_kernel 
fn main() {
    let cuda_files = vec!["src/cuda/testKernel.cu"];
    let bindings_headers = vec!["src/cuda/conv2d.h"];

    // Compile CUDA files
    let mut compiler = cc::Build::new();
    // iterate cuda_files without moving ownership
    for f in &cuda_files {
        compiler.file(f);
    }
    compiler.cuda(true).cudart("static").compile("autograd_cuda");
    // compiler.flag("-arch=sm_60")

    // NOTE bindgen needs clang, delaying this for now better doing it once manually then uploading bindings

    // Generate bindings; for now things are easy enough to do them manually but it's good practice
    // let mut bindings = bindgen::Builder::default();
    // for f in bindings_headers {
    //     bindings = bindings.header(f);
    // }
    
    // let bindings = bindings.parse_callbacks(Box::new(bindgen::CargoCallbacks::new())).generate().expect("Unable to generate bindings");

    // // set before build.rs is executed
    // let out_dir = env::var("OUT_DIR").unwrap();
    // let out_path = PathBuf::from(&out_dir);
    // bindings.write_to_file(out_path.join("bindings.rs"))
    //     .expect("Couldn't write bindings!");


    // Link with CUDA runtime library
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
