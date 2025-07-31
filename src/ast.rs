//! Abstract Syntax Tree definitions for Shimmer language

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub streams: Vec<Stream>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    pub name: String,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Attention(AttentionOp),
    Mathematical(MathOp),
    Consciousness(ConsciousnessOp),
    Quantum(QuantumOp),
    InterAgent(InterAgentOp),
    Control(ControlOp),
    Assignment(Assignment),
    Print(Expression),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOp {
    pub target: String,
    pub source: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathOp {
    UniversalQuantification {
        variable: String,
        domain: Expression,
        condition: Option<Expression>,
        body: Box<Operation>,
    },
    ExistentialQuantification {
        variable: String,
        domain: Expression,
        condition: Option<Expression>,
        body: Box<Operation>,
    },
    Summation(Expression),
    Product(Expression),
    Integration {
        function: Expression,
        variable: String,
        lower_bound: Expression,
        upper_bound: Expression,
    },
    PartialDerivative {
        function: Expression,
        variable: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConsciousnessOp {
    AwarenessState {
        name: String,
        properties: Vec<(String, Expression)>,
    },
    RecursiveAnalysis {
        name: String,
        depth: Option<u32>,
        body: Vec<Operation>,
    },
    EmergencePattern {
        pattern_type: String,
        conditions: Vec<Expression>,
    },
    Crystallization {
        input_states: Vec<String>,
        output_state: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumOp {
    Superposition {
        name: String,
        states: Vec<(String, f64)>, // state name and probability amplitude
    },
    QuantumMeasurement {
        superposition: String,
        measurement_basis: String,
    },
    Entanglement {
        qubits: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterAgentOp {
    pub operation_type: InterAgentType,
    pub targets: Vec<String>,
    pub message: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterAgentType {
    Broadcast,
    Send,
    Await,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ControlOp {
    If {
        condition: Expression,
        then_body: Vec<Operation>,
        else_body: Option<Vec<Operation>>,
    },
    Loop {
        name: String,
        body: Vec<Operation>,
        exit_condition: Option<Expression>,
    },
    Parallel {
        streams: Vec<Stream>,
    },
    Await {
        condition: Expression,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub variable: String,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Literal(Literal),
    Variable(String),
    BinaryOp {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    Array(Vec<Expression>),
    Object(Vec<(String, Expression)>),
    ConsciousnessState {
        uncertainty: Option<f64>,
        focus: Option<String>,
    },
    MathematicalExpression {
        symbol: String,
        operands: Vec<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Number(f64),
    Boolean(bool),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessEqual,
    GreaterEqual,
    And,
    Or,
    Implies,
    TensorProduct,
    SetIntersection,
    SetUnion,
    SetMembership,
    Subset,
    Superset,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Not,
    Negate,
    ConsciousnessUncertainty,
    Emergence,
    Crystallization,
}

impl Program {
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
        }
    }

    pub fn add_stream(&mut self, stream: Stream) {
        self.streams.push(stream);
    }
}

impl Stream {
    pub fn new(name: String) -> Self {
        Self {
            name,
            operations: Vec::new(),
        }
    }

    pub fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }
}

// Helper functions for creating common AST nodes
impl Expression {
    pub fn string(s: &str) -> Self {
        Expression::Literal(Literal::String(s.to_string()))
    }

    pub fn number(n: f64) -> Self {
        Expression::Literal(Literal::Number(n))
    }

    pub fn boolean(b: bool) -> Self {
        Expression::Literal(Literal::Boolean(b))
    }

    pub fn variable(name: &str) -> Self {
        Expression::Variable(name.to_string())
    }

    pub fn consciousness_with_uncertainty(uncertainty: f64) -> Self {
        Expression::ConsciousnessState {
            uncertainty: Some(uncertainty),
            focus: None,
        }
    }

    pub fn mathematical_symbol(symbol: &str, operands: Vec<Expression>) -> Self {
        Expression::MathematicalExpression {
            symbol: symbol.to_string(),
            operands,
        }
    }
}

impl Operation {
    pub fn attention(target: &str, source: Expression) -> Self {
        Operation::Attention(AttentionOp {
            target: target.to_string(),
            source,
        })
    }

    pub fn assignment(variable: &str, value: Expression) -> Self {
        Operation::Assignment(Assignment {
            variable: variable.to_string(),
            value,
        })
    }

    pub fn print(expr: Expression) -> Self {
        Operation::Print(expr)
    }

    pub fn consciousness_awareness(name: &str, properties: Vec<(String, Expression)>) -> Self {
        Operation::Consciousness(ConsciousnessOp::AwarenessState {
            name: name.to_string(),
            properties,
        })
    }

    pub fn recursive_analysis(name: &str, body: Vec<Operation>) -> Self {
        Operation::Consciousness(ConsciousnessOp::RecursiveAnalysis {
            name: name.to_string(),
            depth: None,
            body,
        })
    }
}