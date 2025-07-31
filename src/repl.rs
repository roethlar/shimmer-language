//! Shimmer REPL (Read-Eval-Print Loop)

use shimmer_lang::{ShimmerCompiler, ShimmerConfig};
use std::io::{self, Write};

fn main() {
    println!("🎯 Shimmer Language REPL v0.1.0");
    println!("AI-native programming with consciousness modeling");
    println!("Type 'help' for commands, 'quit' to exit\n");

    let compiler = ShimmerCompiler::new();
    
    loop {
        print!("shimmer> ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();
                
                match input {
                    "quit" | "exit" | "q" => {
                        println!("Goodbye! 🌟");
                        break;
                    }
                    "help" | "h" => {
                        print_help();
                    }
                    "version" | "v" => {
                        println!("Shimmer Language v{}", shimmer_lang::VERSION);
                    }
                    "" => continue,
                    _ => {
                        if input.starts_with("|||") {
                            // Shimmer code
                            match compiler.compile(input) {
                                Ok(rust_code) => {
                                    println!("✅ Compiled successfully:");
                                    println!("{}", rust_code);
                                }
                                Err(e) => {
                                    println!("❌ Compilation error: {}", e);
                                }
                            }
                        } else {
                            // Single expression
                            let wrapped = format!("||| repl_expr |||\n    PRINT {}\n|||", input);
                            match compiler.compile(&wrapped) {
                                Ok(rust_code) => {
                                    println!("✅ Expression compiled:");
                                    println!("{}", rust_code);
                                }
                                Err(e) => {
                                    println!("❌ Expression error: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            Err(error) => {
                println!("❌ Input error: {}", error);
                break;
            }
        }
    }
}

fn print_help() {
    println!(r#"
Shimmer REPL Commands:
  help, h      - Show this help message
  version, v   - Show version information
  quit, exit, q - Exit the REPL

Shimmer Syntax:
  ||| stream_name |||
      ATTN variable → expression
      PRINT expression
      ∀ agent ∈ agents: operation
      ◊ consciousness_state := properties
  |||

Examples:
  PRINT "Hello, Shimmer!"
  ∀ x ∈ [1,2,3]: PRINT x
  ◊ awareness := "focused"

Mathematical Operators:
  ∀  - Universal quantification (for all)
  ∃  - Existential quantification (there exists)
  ∑  - Summation
  ∏  - Product
  ∫  - Integration
  ∂  - Partial derivative

Consciousness Operators:
  ◊  - Consciousness/awareness state
  ⟲  - Recursive analysis
  ⬆  - Emergence pattern
  ⭐  - Crystallization
  ⊕  - Quantum superposition
"#);
}