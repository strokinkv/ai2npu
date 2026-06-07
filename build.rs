use std::path::PathBuf;

fn main() {
    let openvino_libs = std::env::var("OPENVINO_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("OPENVINO_SDK_ROOT")
                .map(PathBuf::from)
                .map(|root| {
                    root.join("runtime")
                        .join("lib")
                        .join("intel64")
                        .join("Release")
                })
        })
        .expect("set OPENVINO_LIB_DIR or OPENVINO_SDK_ROOT");

    println!("cargo:rustc-link-search=native={}", openvino_libs.display());
    println!("cargo:rustc-link-lib=dylib=openvino_c");
    println!("cargo:rerun-if-env-changed=OPENVINO_LIB_DIR");
    println!("cargo:rerun-if-env-changed=OPENVINO_SDK_ROOT");
    println!("cargo:rerun-if-changed=build.rs");
}
