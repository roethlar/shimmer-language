//! Shimmer language parser

use crate::ast::*;
use std::fmt;

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parse error at line {}, column {}: {}", 
               self.line, self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

pub struct ShimmerParser {
    tokens: Vec<Token>,
    current: usize,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Delimiters
    TripleBar,        // |||
    LeftBrace,        // {
    RightBrace,       // }
    LeftParen,        // (
    RightParen,       // )
    LeftBracket,      // [
    RightBracket,     // ]
    
    // Operators
    Arrow,            // →
    Implies,          // ⤏
    TensorProduct,    // ⊗
    
    // Mathematical symbols
    ForAll,           // ∀
    Exists,           // ∃
    Sum,              // ∑
    Product,          // ∏
    Integral,         // ∫
    PartialDerivative,// ∂
    Element,          // ∈
    NotElement,       // ∉
    Subset,           // ⊆
    Superset,         // ⊇
    Intersection,     // ∩
    Union,            // ∪
    
    // Consciousness symbols  
    Consciousness,    // ◊
    Recursive,        // ⟲
    Emergence,        // ⬆
    Crystallization,  // ⭐
    Superposition,    // ⊕
    
    // Keywords
    Attn,
    Print,
    If,
    Else,
    Loop,
    Await,
    InterAgent,
    Broadcast,
    Send,
    
    // Literals
    String(String),
    Number(f64),
    Boolean(bool),
    Identifier(String),
    
    // Punctuation
    Comma,
    Semicolon,
    Colon,
    Assign,           // :=
    
    // Special
    Newline,
    Eof,
}

impl ShimmerParser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            current: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn parse(&mut self, input: &str) -> Result<Program, ParseError> {
        self.tokenize(input)?;
        self.parse_program()
    }

    fn tokenize(&mut self, input: &str) -> Result<(), ParseError> {
        let mut chars = input.chars().peekable();
        self.line = 1;
        self.column = 1;
        
        while let Some(ch) = chars.next() {
            match ch {
                ' ' | '\t' => {
                    self.column += 1;
                }
                '\n' => {
                    self.tokens.push(Token::Newline);
                    self.line += 1;
                    self.column = 1;
                }
                '|' => {
                    if chars.peek() == Some(&'|') {
                        chars.next(); // consume second |
                        if chars.peek() == Some(&'|') {
                            chars.next(); // consume third |
                            self.tokens.push(Token::TripleBar);
                            self.column += 3;
                        } else {
                            return Err(ParseError {
                                message: "Expected '|||' but found '||'".to_string(),
                                line: self.line,
                                column: self.column,
                            });
                        }
                    }
                }
                '{' => {
                    self.tokens.push(Token::LeftBrace);
                    self.column += 1;
                }
                '}' => {
                    self.tokens.push(Token::RightBrace);
                    self.column += 1;
                }
                '(' => {
                    self.tokens.push(Token::LeftParen);
                    self.column += 1;
                }
                ')' => {
                    self.tokens.push(Token::RightParen);
                    self.column += 1;
                }
                '[' => {
                    self.tokens.push(Token::LeftBracket);
                    self.column += 1;
                }
                ']' => {
                    self.tokens.push(Token::RightBracket);
                    self.column += 1;
                }
                ',' => {
                    self.tokens.push(Token::Comma);
                    self.column += 1;
                }
                ';' => {
                    self.tokens.push(Token::Semicolon);
                    self.column += 1;
                }
                ':' => {
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Assign);
                        self.column += 2;
                    } else {
                        self.tokens.push(Token::Colon);
                        self.column += 1;
                    }
                }
                '"' => {
                    let string_value = self.parse_string(&mut chars)?;
                    self.tokens.push(Token::String(string_value));
                }
                // Mathematical symbols
                '∀' => {
                    self.tokens.push(Token::ForAll);
                    self.column += 1;
                }
                '∃' => {
                    self.tokens.push(Token::Exists);
                    self.column += 1;
                }
                '∑' => {
                    self.tokens.push(Token::Sum);
                    self.column += 1;
                }
                '∏' => {
                    self.tokens.push(Token::Product);
                    self.column += 1;
                }
                '∫' => {
                    self.tokens.push(Token::Integral);
                    self.column += 1;
                }
                '∂' => {
                    self.tokens.push(Token::PartialDerivative);
                    self.column += 1;
                }
                '∈' => {
                    self.tokens.push(Token::Element);
                    self.column += 1;
                }
                '∉' => {
                    self.tokens.push(Token::NotElement);
                    self.column += 1;
                }
                '⊆' => {
                    self.tokens.push(Token::Subset);
                    self.column += 1;
                }
                '⊇' => {
                    self.tokens.push(Token::Superset);
                    self.column += 1;
                }
                '∩' => {
                    self.tokens.push(Token::Intersection);
                    self.column += 1;
                }
                '∪' => {
                    self.tokens.push(Token::Union);
                    self.column += 1;
                }
                '→' => {
                    self.tokens.push(Token::Arrow);
                    self.column += 1;
                }
                '⤏' => {
                    self.tokens.push(Token::Implies);
                    self.column += 1;
                }
                '⊗' => {
                    self.tokens.push(Token::TensorProduct);
                    self.column += 1;
                }
                // Consciousness symbols
                '◊' => {
                    self.tokens.push(Token::Consciousness);
                    self.column += 1;
                }
                '⟲' => {
                    self.tokens.push(Token::Recursive);
                    self.column += 1;
                }
                '⬆' => {
                    self.tokens.push(Token::Emergence);
                    self.column += 1;
                }
                '⭐' => {
                    self.tokens.push(Token::Crystallization);
                    self.column += 1;
                }
                '⊕' => {
                    self.tokens.push(Token::Superposition);
                    self.column += 1;
                }
                _ => {
                    if ch.is_ascii_alphabetic() || ch == '_' {
                        let identifier = self.parse_identifier(ch, &mut chars);
                        self.tokens.push(self.classify_identifier(&identifier));
                    } else if ch.is_ascii_digit() || ch == '.' {
                        let number = self.parse_number(ch, &mut chars)?;
                        self.tokens.push(Token::Number(number));
                    } else {
                        return Err(ParseError {
                            message: format!("Unexpected character: '{}'", ch),
                            line: self.line,
                            column: self.column,
                        });
                    }
                }
            }
        }
        
        self.tokens.push(Token::Eof);
        Ok(())
    }

    fn parse_string(&mut self, chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<String, ParseError> {
        let mut value = String::new();
        self.column += 1; // for opening quote
        
        while let Some(ch) = chars.next() {
            self.column += 1;
            match ch {
                '"' => return Ok(value),
                '\\' => {
                    if let Some(escaped) = chars.next() {
                        match escaped {
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            'r' => value.push('\r'),
                            '\\' => value.push('\\'),
                            '"' => value.push('"'),
                            _ => {
                                value.push('\\');
                                value.push(escaped);
                            }
                        }
                        self.column += 1;
                    }
                }
                _ => value.push(ch),
            }
        }
        
        Err(ParseError {
            message: "Unterminated string literal".to_string(),
            line: self.line,
            column: self.column,
        })
    }

    fn parse_identifier(&mut self, first_char: char, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut identifier = String::new();
        identifier.push(first_char);
        self.column += 1;
        
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                identifier.push(ch);
                chars.next();
                self.column += 1;
            } else {
                break;
            }
        }
        
        identifier
    }

    fn parse_number(&mut self, first_char: char, chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<f64, ParseError> {
        let mut number_str = String::new();
        number_str.push(first_char);
        self.column += 1;
        
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                number_str.push(ch);
                chars.next();
                self.column += 1;
            } else {
                break;
            }
        }
        
        number_str.parse().map_err(|_| ParseError {
            message: format!("Invalid number: {}", number_str),
            line: self.line,
            column: self.column,
        })
    }

    fn classify_identifier(&self, identifier: &str) -> Token {
        match identifier {
            "ATTN" => Token::Attn,
            "PRINT" => Token::Print,
            "IF" => Token::If,
            "ELSE" => Token::Else,
            "LOOP" => Token::Loop,
            "AWAIT" => Token::Await,
            "INTER_AGENT" => Token::InterAgent,
            "BROADCAST" => Token::Broadcast,
            "SEND" => Token::Send,
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            _ => Token::Identifier(identifier.to_string()),
        }
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut program = Program::new();
        
        while !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            
            let stream = self.parse_stream()?;
            program.add_stream(stream);
        }
        
        Ok(program)
    }

    fn parse_stream(&mut self) -> Result<Stream, ParseError> {
        self.consume(&Token::TripleBar, "Expected '|||' at start of stream")?;
        
        let name = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err(ParseError {
                message: "Expected stream name".to_string(),
                line: self.line,
                column: self.column,
            }),
        };
        
        self.consume(&Token::TripleBar, "Expected '|||' after stream name")?;
        
        let mut stream = Stream::new(name);
        
        while !self.check(&Token::TripleBar) && !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            
            let operation = self.parse_operation()?;
            stream.add_operation(operation);
        }
        
        self.consume(&Token::TripleBar, "Expected '|||' at end of stream")?;
        
        Ok(stream)
    }

    fn parse_operation(&mut self) -> Result<Operation, ParseError> {
        match self.peek() {
            Token::Attn => self.parse_attention(),
            Token::Print => self.parse_print(),
            Token::ForAll => self.parse_universal_quantification(),
            Token::Exists => self.parse_existential_quantification(),
            Token::Consciousness => self.parse_consciousness_operation(),
            Token::Recursive => self.parse_recursive_analysis(),
            Token::Identifier(_) => self.parse_assignment_or_expression(),
            _ => Err(ParseError {
                message: format!("Unexpected token: {:?}", self.peek()),
                line: self.line,
                column: self.column,
            }),
        }
    }

    fn parse_attention(&mut self) -> Result<Operation, ParseError> {
        self.advance(); // consume ATTN
        
        let target = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err(ParseError {
                message: "Expected identifier after ATTN".to_string(),
                line: self.line,
                column: self.column,
            }),
        };
        
        self.consume(&Token::Arrow, "Expected '→' after ATTN target")?;
        
        let source = self.parse_expression()?;
        
        Ok(Operation::attention(&target, source))
    }

    fn parse_print(&mut self) -> Result<Operation, ParseError> {
        self.advance(); // consume PRINT
        let expr = self.parse_expression()?;
        Ok(Operation::print(expr))
    }

    fn parse_universal_quantification(&mut self) -> Result<Operation, ParseError> {
        self.advance(); // consume ∀
        
        let variable = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err(ParseError {
                message: "Expected variable after ∀".to_string(),
                line: self.line,
                column: self.column,
            }),
        };
        
        self.consume(&Token::Element, "Expected '∈' after universal quantification variable")?;
        
        let domain = self.parse_expression()?;
        
        // Optional condition
        let condition = if self.match_token(&Token::Colon) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        let body = Box::new(self.parse_operation()?);
        
        Ok(Operation::Mathematical(MathOp::UniversalQuantification {
            variable,
            domain,
            condition,
            body,
        }))
    }

    fn parse_existential_quantification(&mut self) -> Result<Operation, ParseError> {
        self.advance(); // consume ∃
        
        let variable = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err(ParseError {
                message: "Expected variable after ∃".to_string(),
                line: self.line,
                column: self.column,
            }),
        };
        
        self.consume(&Token::Element, "Expected '∈' after existential quantification variable")?;
        
        let domain = self.parse_expression()?;
        
        let condition = if self.match_token(&Token::Colon) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        let body = Box::new(self.parse_operation()?);
        
        Ok(Operation::Mathematical(MathOp::ExistentialQuantification {
            variable,
            domain,
            condition,
            body,
        }))
    }

    fn parse_consciousness_operation(&mut self) -> Result<Operation, ParseError> {
        self.advance(); // consume ◊
        
        let name = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err(ParseError {
                message: "Expected identifier after ◊".to_string(),
                line: self.line,
                column: self.column,
            }),
        };
        
        self.consume(&Token::Assign, "Expected ':=' after consciousness state name")?;
        
        // Parse consciousness state properties
        if self.check(&Token::LeftBrace) {
            self.advance(); // consume {
            let mut properties = Vec::new();
            
            while !self.check(&Token::RightBrace) && !self.is_at_end() {
                let prop_name = match self.advance() {
                    Token::Identifier(name) => name,
                    _ => return Err(ParseError {
                        message: "Expected property name".to_string(),
                        line: self.line,
                        column: self.column,
                    }),
                };
                
                self.consume(&Token::Colon, "Expected ':' after property name")?;
                let value = self.parse_expression()?;
                properties.push((prop_name, value));
                
                if !self.check(&Token::RightBrace) {
                    self.consume(&Token::Comma, "Expected ',' between properties")?;
                }
            }
            
            self.consume(&Token::RightBrace, "Expected '}' after consciousness properties")?;
            
            Ok(Operation::consciousness_awareness(&name, properties))
        } else {
            let value = self.parse_expression()?;
            Ok(Operation::assignment(&name, value))
        }
    }

    fn parse_recursive_analysis(&mut self) -> Result<Operation, ParseError> {
        self.advance(); // consume ⟲
        
        let name = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err(ParseError {
                message: "Expected identifier after ⟲".to_string(),
                line: self.line,
                column: self.column,
            }),
        };
        
        self.consume(&Token::Assign, "Expected ':=' after recursive analysis name")?;
        self.consume(&Token::LeftBrace, "Expected '{' after recursive analysis assignment")?;
        
        let mut body = Vec::new();
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            body.push(self.parse_operation()?);
        }
        
        self.consume(&Token::RightBrace, "Expected '}' after recursive analysis body")?;
        
        Ok(Operation::recursive_analysis(&name, body))
    }

    fn parse_assignment_or_expression(&mut self) -> Result<Operation, ParseError> {
        let checkpoint = self.current;
        
        if let Token::Identifier(name) = self.advance() {
            if self.match_token(&Token::Assign) {
                let value = self.parse_expression()?;
                return Ok(Operation::assignment(&name, value));
            }
        }
        
        // Reset and parse as expression
        self.current = checkpoint;
        let expr = self.parse_expression()?;
        Ok(Operation::Print(expr)) // Default to printing expressions
    }

    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_binary_expression()
    }

    fn parse_binary_expression(&mut self) -> Result<Expression, ParseError> {
        let mut expr = self.parse_primary_expression()?;
        
        while let Some(op) = self.parse_binary_operator() {
            let right = self.parse_primary_expression()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_binary_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Token::Arrow => {
                self.advance();
                Some(BinaryOperator::Implies)
            }
            Token::TensorProduct => {
                self.advance();
                Some(BinaryOperator::TensorProduct)
            }
            Token::Element => {
                self.advance();
                Some(BinaryOperator::SetMembership)
            }
            Token::Intersection => {
                self.advance();
                Some(BinaryOperator::SetIntersection)
            }
            Token::Union => {
                self.advance();
                Some(BinaryOperator::SetUnion)
            }
            _ => None,
        }
    }

    fn parse_primary_expression(&mut self) -> Result<Expression, ParseError> {
        match self.advance() {
            Token::String(s) => Ok(Expression::string(&s)),
            Token::Number(n) => Ok(Expression::number(n)),
            Token::Boolean(b) => Ok(Expression::boolean(b)),
            Token::Identifier(name) => Ok(Expression::variable(&name)),
            Token::Consciousness => {
                // Consciousness state with optional uncertainty
                if self.match_token(&Token::String(String::new())) {
                    if let Token::String(focus) = self.previous() {
                        Ok(Expression::ConsciousnessState {
                            uncertainty: None,
                            focus: Some(focus.to_string()),
                        })
                    } else {
                        Ok(Expression::ConsciousnessState {
                            uncertainty: None,
                            focus: None,
                        })
                    }
                } else {
                    Ok(Expression::ConsciousnessState {
                        uncertainty: None,
                        focus: None,
                    })
                }
            }
            Token::LeftBracket => {
                let mut elements = Vec::new();
                
                while !self.check(&Token::RightBracket) && !self.is_at_end() {
                    elements.push(self.parse_expression()?);
                    if !self.check(&Token::RightBracket) {
                        self.consume(&Token::Comma, "Expected ',' between array elements")?;
                    }
                }
                
                self.consume(&Token::RightBracket, "Expected ']' after array elements")?;
                Ok(Expression::Array(elements))
            }
            _ => Err(ParseError {
                message: format!("Unexpected token in expression: {:?}", self.previous()),
                line: self.line,
                column: self.column,
            }),
        }
    }

    // Helper methods
    fn match_token(&mut self, expected: &Token) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, expected: &Token) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(self.peek()) == std::mem::discriminant(expected)
        }
    }

    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous().clone()
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn consume(&mut self, expected: &Token, error_message: &str) -> Result<Token, ParseError> {
        if self.check(expected) {
            Ok(self.advance())
        } else {
            Err(ParseError {
                message: error_message.to_string(),
                line: self.line,
                column: self.column,
            })
        }
    }
}