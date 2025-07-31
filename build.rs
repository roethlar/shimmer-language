// Build script for Shimmer Language
// Handles protocol buffer compilation and GPU runtime setup


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile protocol buffers
    let proto_files = vec![
        "protos/shimmer_types.proto",
        "protos/gpu_acceleration.proto",
    ];
    
    let proto_include_dir = "protos";
    
    println!("cargo:rerun-if-changed={}", proto_include_dir);
    
    for proto_file in &proto_files {
        println!("cargo:rerun-if-changed={}", proto_file);
    }
    
    // Configure tonic-build for protocol buffer compilation
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")
        .compile(&proto_files, &[proto_include_dir])?;
    
    // Create generated directory if it doesn't exist
    std::fs::create_dir_all("src/generated")?;
    
    // GPU runtime configuration
    configure_gpu_runtime()?;
    
    Ok(())
}

fn configure_gpu_runtime() -> Result<(), Box<dyn std::error::Error>> {
    // Check for CUDA availability
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("Found CUDA at: {}", cuda_path);
    }
    
    // Check for ROCm (AMD GPU) availability
    if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", rocm_path);
        println!("cargo:rustc-link-lib=hip");
        println!("cargo:rustc-cfg=feature=\"rocm\"");
        println!("Found ROCm at: {}", rocm_path);
    }
    
    // macOS Metal support (always available on macOS)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-cfg=feature=\"metal\"");
        println!("Metal support enabled for macOS");
    }
    
    Ok(())
}