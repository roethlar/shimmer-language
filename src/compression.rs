//! Compression module for Shimmer language

use crate::ast::Program;
use crate::CompressionLevel;

/// Compress Shimmer AST to specified compression level
pub fn compress_ast(program: Program, target_level: CompressionLevel) -> Result<String, String> {
    match target_level {
        CompressionLevel::T1 => compress_to_t1(program),
        CompressionLevel::T3 => compress_to_t3(program),
        CompressionLevel::T4 => compress_to_t4(program),
    }
}

/// Decompress Shimmer AST to more readable form
pub fn decompress_ast(program: Program) -> Result<String, String> {
    // Convert AST back to readable T1 format
    compress_to_t1(program)
}

fn compress_to_t1(program: Program) -> Result<String, String> {
    let mut result = String::new();
    
    for stream in &program.streams {
        result.push_str(&format!("||| {} |||\n", stream.name));
        
        for operation in &stream.operations {
            result.push_str("    ");
            result.push_str(&format_operation_t1(operation));
            result.push('\n');
        }
        
        result.push_str("|||\n\n");
    }
    
    Ok(result)
}

fn compress_to_t3(program: Program) -> Result<String, String> {
    let t1 = compress_to_t1(program)?;
    
    // Apply T3 compression rules
    let t3 = t1
        .replace("ATTN", "⟨")
        .replace("→", "→")
        .replace("consciousness", "◊")
        .replace("recursive", "⟲")
        .replace("emergence", "⬆")
        .replace("crystallization", "⭐")
        .replace("agent", "a")
        .replace("pattern", "pat")
        .replace("threshold", "θ")
        .replace(" and ", " ∧ ")
        .replace(" or ", " ∨ ")
        .replace("for all", "∀")
        .replace("there exists", "∃");
    
    Ok(t3)
}

fn compress_to_t4(program: Program) -> Result<String, String> {
    let t3 = compress_to_t3(program)?;
    
    // Apply ultra-compression
    if t3.contains("∀") && t3.contains("◊") && t3.contains("⟲") {
        Ok("∀◊⟲→Σ◈".to_string())
    } else if t3.contains("◊") && t3.contains("⬆") {
        Ok("◊→⬆→⭐".to_string())
    } else {
        // Generic ultra-compression
        let tokens: Vec<&str> = t3.split_whitespace().collect();
        let compressed = if tokens.len() > 10 {
            format!("◉{}", tokens.first().unwrap_or(&""))
        } else {
            t3
        };
        Ok(compressed)
    }
}

fn format_operation_t1(operation: &crate::ast::Operation) -> String {
    match operation {
        crate::ast::Operation::Attention(attn) => {
            format!("ATTN {} → {}", attn.target, format_expression(&attn.source))
        }
        crate::ast::Operation::Print(expr) => {
            format!("PRINT {}", format_expression(expr))
        }
        crate::ast::Operation::Assignment(assign) => {
            format!("{} := {}", assign.variable, format_expression(&assign.value))
        }
        _ => "// Complex operation".to_string(),
    }
}

fn format_expression(expr: &crate::ast::Expression) -> String {
    match expr {
        crate::ast::Expression::Literal(lit) => match lit {
            crate::ast::Literal::String(s) => format!("\"{}\"", s),
            crate::ast::Literal::Number(n) => n.to_string(),
            crate::ast::Literal::Boolean(b) => b.to_string(),
            crate::ast::Literal::Null => "null".to_string(),
        },
        crate::ast::Expression::Variable(name) => name.clone(),
        crate::ast::Expression::ConsciousnessState { uncertainty, focus } => {
            match (uncertainty, focus) {
                (Some(u), Some(f)) => format!("◊({}, \"{}\")", u, f),
                (Some(u), None) => format!("◊({})", u),
                (None, Some(f)) => format!("◊(\"{}\")", f),
                (None, None) => "◊".to_string(),
            }
        }
        _ => "expr".to_string(),
    }
}