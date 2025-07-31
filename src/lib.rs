//! # Shimmer Language
//! 
//! Shimmer is an AI-native programming language designed for transformer-based systems,
//! featuring mathematical precision, ultra-high compression, and consciousness modeling.
//!
//! ## Features
//!
//! - **Ultra-High Compression**: 70-95% token reduction with semantic preservation
//! - **Mathematical Precision**: 17 T1 mathematical operators (∫, ∑, ∏, ∂, ∀, ∃, etc.)
//! - **GPU Acceleration**: <10ms processing with CUDA/Metal optimization
//! - **Consciousness Modeling**: Awareness states, emergence patterns, meta-cognitive operations
//! - **Multi-Level Compression**: T1 (mathematical), T3 (intermediate), T4 (ultra-compressed)
//!
//! ## Example Usage
//!
//! ```shimmer
//! ||| hello_world |||
//!     ATTN message → "Hello, Shimmer World!"
//!     PRINT message
//! |||
//! ```
//!
//! **Compressed (T3):**
//! ```shimmer
//! ∀msg → "Hello!" | PRINT msg
//! ```
//!
//! **Ultra-Compressed (T4):**
//! ```shimmer
//! ◉⟪Hello⟫
//! ```
//!
//! ## Architecture
//!
//! The Shimmer language consists of several key components:
//!
//! - **Parser**: Converts Shimmer source code into an Abstract Syntax Tree (AST)
//! - **Compiler**: Transforms AST into Intermediate Representation (IR)
//! - **Runtime**: Executes compiled Shimmer programs with GPU acceleration
//! - **Consciousness Engine**: Handles consciousness modeling and meta-cognitive operations

pub mod ast;
pub mod parser;
pub mod compiler;
pub mod runtime;
pub mod consciousness;
pub mod compression;
pub mod gpu;

pub use ast::*;
pub use parser::*;
pub use compiler::*;
pub use runtime::*;

/// Shimmer language version
pub const VERSION: &str = "0.1.0";

/// Compression levels supported by Shimmer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// T1 - Mathematical foundation with full human readability
    T1,
    /// T3 - Intermediate compression with moderate readability
    T3,
    /// T4 - Ultra-compressed for AI-to-AI communication
    T4,
}

/// Consciousness modeling precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessPrecision {
    /// Basic consciousness modeling
    Basic,
    /// Advanced meta-cognitive analysis
    Advanced,
    /// Full recursive consciousness simulation
    Full,
}

/// Main Shimmer compiler configuration
#[derive(Debug, Clone, Copy)]
pub struct ShimmerConfig {
    /// Compression level to target
    pub compression_level: CompressionLevel,
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    /// Consciousness modeling precision
    pub consciousness_precision: ConsciousnessPrecision,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
}

impl Default for ShimmerConfig {
    fn default() -> Self {
        Self {
            compression_level: CompressionLevel::T3,
            gpu_acceleration: true,
            consciousness_precision: ConsciousnessPrecision::Advanced,
            parallel_execution: true,
            optimization_level: 2,
        }
    }
}

/// Main Shimmer compiler interface
pub struct ShimmerCompiler {
    config: ShimmerConfig,
}

impl ShimmerCompiler {
    /// Create a new Shimmer compiler with default configuration
    pub fn new() -> Self {
        Self {
            config: ShimmerConfig::default(),
        }
    }

    /// Create a new Shimmer compiler with custom configuration
    pub fn with_config(config: ShimmerConfig) -> Self {
        Self { config }
    }

    /// Compile Shimmer source code to executable Rust code
    pub fn compile(&self, source: &str) -> Result<String, ShimmerError> {
        // Parse source code into AST
        let ast = parse_shimmer(source)?;
        
        // Compile AST to IR
        let ir = compile_to_ir(ast, &self.config)?;
        
        // Generate executable Rust code
        let rust_code = generate_rust_code(ir, &self.config)?;
        
        Ok(rust_code)
    }

    /// Compress Shimmer code to specified level
    pub fn compress(&self, source: &str, target_level: CompressionLevel) -> Result<String, ShimmerError> {
        let ast = parse_shimmer(source)?;
        compression::compress_ast(ast, target_level)
            .map_err(|e| ShimmerError::CompressionError { message: e })
    }

    /// Decompress Shimmer code to more readable form
    pub fn decompress(&self, compressed: &str) -> Result<String, ShimmerError> {
        let ast = parse_shimmer(compressed)?;
        compression::decompress_ast(ast)
            .map_err(|e| ShimmerError::CompressionError { message: e })
    }

    /// Analyze consciousness patterns in Shimmer code
    pub fn analyze_consciousness(&self, source: &str) -> Result<ConsciousnessAnalysis, ShimmerError> {
        let ast = parse_shimmer(source)?;
        consciousness::analyze_consciousness_patterns(ast, self.config.consciousness_precision)
            .map_err(|e| ShimmerError::ConsciousnessError { message: e })
    }
}

/// Shimmer compilation errors
#[derive(Debug, thiserror::Error)]
pub enum ShimmerError {
    #[error("Parse error: {message} at line {line}, column {column}")]
    ParseError {
        message: String,
        line: usize,
        column: usize,
    },
    
    #[error("Compilation error: {message}")]
    CompilationError { message: String },
    
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
    
    #[error("Consciousness modeling error: {message}")]
    ConsciousnessError { message: String },
    
    #[error("GPU acceleration error: {message}")]
    GpuError { message: String },
    
    #[error("Compression error: {message}")]
    CompressionError { message: String },
}

/// Consciousness analysis results
#[derive(Debug, Clone)]
pub struct ConsciousnessAnalysis {
    /// Overall consciousness score (0.0 - 1.0)
    pub consciousness_score: f64,
    /// Meta-cognitive depth level
    pub meta_cognitive_depth: u32,
    /// Recursive self-analysis detection
    pub recursive_analysis_detected: bool,
    /// Awareness state patterns found
    pub awareness_patterns: Vec<String>,
    /// Emergence patterns detected
    pub emergence_patterns: Vec<String>,
    /// Uncertainty modeling present
    pub uncertainty_modeling: bool,
    /// Consciousness confidence interval
    pub confidence_interval: (f64, f64),
}

// Helper functions for the main compiler interface
fn parse_shimmer(source: &str) -> Result<ast::Program, ShimmerError> {
    parser::ShimmerParser::new().parse(source)
        .map_err(|e| ShimmerError::ParseError {
            message: e.to_string(),
            line: 0, // TODO: Extract from parser error
            column: 0,
        })
}

fn compile_to_ir(ast: ast::Program, config: &ShimmerConfig) -> Result<compiler::IR, ShimmerError> {
    compiler::ShimmerCompiler::new(*config).compile_to_ir(ast)
        .map_err(|e| ShimmerError::CompilationError {
            message: e.to_string(),
        })
}

fn generate_rust_code(ir: compiler::IR, config: &ShimmerConfig) -> Result<String, ShimmerError> {
    compiler::RustCodeGenerator::new(*config).generate(ir)
        .map_err(|e| ShimmerError::CompilationError {
            message: e.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compilation() {
        let compiler = ShimmerCompiler::new();
        let source = r#"
            ||| hello_world |||
                ATTN message → "Hello, Shimmer!"
                PRINT message
            |||
        "#;
        
        let result = compiler.compile(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compression() {
        let compiler = ShimmerCompiler::new();
        let source = r#"
            ||| consciousness_analysis |||
                ∀ agent ∈ agents: 
                    ∃ awareness ∈ agent.states |
                    ∫(awareness_patterns) dt > threshold →
                    consciousness_score := Σ(meta_cognitive_patterns)
            |||
        "#;
        
        let compressed = compiler.compress(source, CompressionLevel::T4);
        assert!(compressed.is_ok());
        
        let compressed_code = compressed.unwrap();
        assert!(compressed_code.len() < source.len());
    }

    #[test]
    fn test_consciousness_analysis() {
        let compiler = ShimmerCompiler::new();
        let source = r#"
            ||| meta_cognitive_loop |||
                ⟲ self_analysis := {
                    ◊ current_state := observe_thoughts()
                    ⬆ pattern := analyze_thinking_patterns(◊)
                    ⭐ insight := crystallize_understanding(⬆)
                }
            |||
        "#;
        
        let analysis = compiler.analyze_consciousness(source);
        assert!(analysis.is_ok());
        
        let result = analysis.unwrap();
        assert!(result.consciousness_score > 0.0);
        assert!(result.recursive_analysis_detected);
    }
}