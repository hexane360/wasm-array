use core::num::{ParseFloatError, ParseIntError};
use std::str::FromStr;
use std::rc::Rc;
use std::collections::HashMap;

use num::Complex;
use itertools::Itertools;
use logos::Logos;

use arraylib::array::DynArray;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\r\n\f]+", error = ParseError)]
#[logos(subpattern exp = r"[eE][+-]?[0-9]+")]
#[logos(subpattern dec = r"[0-9][0-9_]*")]
pub enum Token {
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Times,
    #[token("/")]
    Divide,
    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
    #[token(",")]
    Comma,
    #[token("**")]
    Power,
    #[token("@")]
    At,
    #[token("%")]
    Percent,
    #[token("pi")]
    Pi,
    #[token("&")]
    BitAnd,
    #[token("|")]
    BitOr,
    #[token("^")]
    BitXor,
    #[token("==")]
    Eq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    LessThanEq,
    #[token("<")]
    LessThan,
    #[token(">=")]
    GreaterThanEq,
    #[token(">")]
    GreaterThan,
    #[token("!")]
    Not,

    #[regex(r"(?&dec)", parse_int)]
    IntLit(i64),
    // inf, nan
    #[regex(r"(inf|nan)", parse_float, ignore(ascii_case))]
    // 5., 5.e3
    #[regex(r"(?&dec)\.(?&dec)?(?&exp)?", parse_float)]
    // .5, .5e-3
    #[regex(r"\.(?&dec)(?&exp)?", parse_float)]
    // 1e5
    #[regex(r"(?&dec)(?&exp)", parse_float)]
    FloatLit(f64),
    // 5j, 5.j, 5.5j, 5e3j
    #[regex(r"(?&dec)(\.(?&dec)?)?(?&exp)?j", parse_complex)]
    // .5j, .5e-3j
    #[regex(r"\.(?&dec)(?&exp)?j", parse_complex)]
    ComplexLit(Complex<f64>),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9]*", |lexer| Rc::from(lexer.slice()))]
    Ident(Rc<str>),
    ArrayLit(DynArray),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9]*\(", |lexer| Rc::from(lexer.slice().strip_suffix('(').unwrap()))]
    FuncOpen(Rc<str>),
}

fn parse_complex<'a>(lexer: &mut logos::Lexer<'a, Token>) -> Result<Complex<f64>, ParseFloatError> {
    let s = lexer.slice();
    f64::from_str(&s[..s.len()-1].replace("_", "")).map(|v| Complex::new(0., v))
}

fn parse_float<'a>(lexer: &mut logos::Lexer<'a, Token>) -> Result<f64, ParseFloatError> {
    f64::from_str(&lexer.slice().replace("_", ""))
}

fn parse_int<'a>(lexer: &mut logos::Lexer<'a, Token>) -> Result<i64, ParseIntError> {
    i64::from_str_radix(&lexer.slice().replace("_", ""), 10)
}

pub struct Lexer<I> {
    pub inner: I, //logos::Lexer<'a, Token>,
    // outer option: presence of peek
    // inner option: presence of token/eof
    peek: Option<Option<Token>>,
}

impl<I: Iterator<Item = Result<Token, ParseError>>> Lexer<I> {
    pub fn next(&mut self) {
        self.peek = None;
    }

    pub fn peek(&mut self) -> Result<Option<Token>, ParseError>  {
        if let Some(s) = &self.peek {
            return Ok(s.clone());
        }
        let next = self.inner.next().transpose()?;
        self.peek = Some(next.clone());
        Ok(next)
    }

    pub fn as_binary_op(&mut self) -> Result<Option<BinaryOp>, ParseError> {
        Ok(self.peek()?.and_then(|token| match token {
            Token::Plus => Some(BinaryOp::Add),
            Token::Minus => Some(BinaryOp::Sub),
            Token::Times => Some(BinaryOp::Mul),
            Token::Divide => Some(BinaryOp::Div),

            Token::Percent => Some(BinaryOp::Rem),
            Token::Power => Some(BinaryOp::Pow),
            Token::At => Some(BinaryOp::MatMul),

            Token::Eq => Some(BinaryOp::Eq),
            Token::NotEq => Some(BinaryOp::NotEq),
            Token::GreaterThan => Some(BinaryOp::GreaterThan),
            Token::GreaterThanEq => Some(BinaryOp::GreaterThanEq),
            Token::LessThan => Some(BinaryOp::LessThan),
            Token::LessThanEq => Some(BinaryOp::LessThanEq),

            Token::BitAnd => Some(BinaryOp::BitAnd),
            Token::BitOr => Some(BinaryOp::BitOr),

            _ => None,
        }))
    }

    pub fn as_unary_op(&mut self) -> Result<Option<UnaryOp>, ParseError> {
        Ok(self.peek()?.and_then(|token| match token {
            Token::Plus => Some(UnaryOp::Pos),
            Token::Minus => Some(UnaryOp::Neg),
            Token::Not => Some(UnaryOp::Not),
            _ => None,
        }))
    }

    pub fn as_terminal(&mut self) -> Result<Option<Expr>, ParseError> {
        Ok(match self.peek()? {
            Some(Token::IntLit(value)) => Some(Expr::Literal(LiteralExpr::Int(value))),
            Some(Token::FloatLit(value)) => Some(Expr::Literal(LiteralExpr::Float(value))),
            Some(Token::ComplexLit(value)) => Some(Expr::Literal(LiteralExpr::Complex(value))),
            // TODO handle f32/f64 consts here
            Some(Token::Pi) => Some(Expr::Const("pi", std::f64::consts::PI)),
            Some(Token::ArrayLit(value)) => Some(Expr::Literal(LiteralExpr::Array(value.into()))),
            Some(Token::Ident(variable)) => Some(Expr::Variable(VariableExpr(variable.into()))),
            Some(_) => None,
            None => None,
        })
    }
}

impl<I: Iterator<Item = Result<Token, ParseError>>> From<I> for Lexer<I> {
    fn from(value: I) -> Self {
        Lexer {
            inner: value,
            peek: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Unary(UnaryExpr),
    Binary(BinaryExpr),
    Literal(LiteralExpr),
    Const(&'static str, f64),
    Variable(VariableExpr),
    Parenthesized(Box<Expr>),
    FuncCall(FuncExpr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    MatMul,
    Eq,
    NotEq,
    LessThan,
    GreaterThan,
    LessThanEq,
    GreaterThanEq,
    BitAnd,
    BitOr,
    BitXor,
}

impl BinaryOp {
    pub fn precedence(&self) -> i64 {
        match self {
            BinaryOp::BitOr => 2,
            BinaryOp::BitXor => 3,
            BinaryOp::BitAnd => 4,
            BinaryOp::Eq | BinaryOp::NotEq |
                BinaryOp::LessThan | BinaryOp::GreaterThan |
                BinaryOp::LessThanEq | BinaryOp::GreaterThanEq => 5,
            BinaryOp::Add | BinaryOp::Sub => 6,
            BinaryOp::Rem => 7,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::MatMul => 8,
            BinaryOp::Pow => 9,
        }
    }

    pub fn right_assoc(&self) -> bool {
        match self {
            BinaryOp::Pow => true,
            _ => false,
        }
    }

    pub fn precedes(&self, other: &BinaryOp) -> bool {
        if self.precedence() == other.precedence() {
            // `other` is considered the left operator, and `self` the right operator
            self.right_assoc()
        } else {
            self.precedence() > other.precedence()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub inner: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpr {
    pub lhs: Box<Expr>,
    pub op: BinaryOp,
    pub rhs: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralExpr {
    Int(i64),
    Float(f64),
    Complex(Complex<f64>),
    Array(Rc<DynArray>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncExpr {
    pub name: Rc<str>,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableExpr(pub Rc<str>);


#[derive(Debug, Clone, PartialEq)]
pub enum ExecError {
    Other(String),
}

pub trait ArrayFunc {
    fn name(&self) -> &'static str;
    fn call(&self, inputs: &[DynArray]) -> Result<DynArray,  ExecError>;
}

pub struct UnaryFunc<F: Fn(&DynArray) -> DynArray> {
    inner: F,
    name: &'static str,
}

impl<F: Fn(&DynArray) -> DynArray> UnaryFunc<F> {
    pub fn new(name: &'static str, f: F) -> Self { Self { inner: f, name }}
}

impl<F: Fn(&DynArray) -> DynArray> ArrayFunc for UnaryFunc<F> {
    fn name(&self) -> &'static str { self.name }

    fn call(&self, inputs: &[DynArray]) -> Result<DynArray,  ExecError> {
        if inputs.len() != 1 { return Err(ExecError::Other(format!("Function '{}' must be called with one argument", self.name))) }
        Ok((self.inner)(&inputs[0]))
    }
}

pub struct BinaryFunc<F: Fn(&DynArray, &DynArray) -> DynArray> {
    inner: F,
    name: &'static str,
}

impl<F: Fn(&DynArray, &DynArray) -> DynArray> BinaryFunc<F> {
    pub fn new(name: &'static str, f: F) -> Self { Self { inner: f, name }}
}

impl<F: Fn(&DynArray, &DynArray) -> DynArray> ArrayFunc for BinaryFunc<F> {
    fn name(&self) -> &'static str { self.name }

    fn call(&self, inputs: &[DynArray]) -> Result<DynArray,  ExecError> {
        if inputs.len() != 2 { return Err(ExecError::Other(format!("Function '{}' must be called with two arguments", self.name))) }
        Ok((self.inner)(&inputs[0], &inputs[1]))
    }
}

pub type FuncMap = HashMap<&'static str, Box<dyn ArrayFunc + Sync + Send>>;

impl Expr {
    pub fn exec(&self, vars: &HashMap<Rc<str>, DynArray>, funcs: &FuncMap) -> Result<DynArray, ExecError> {
        //log::log(&format!("Expr::exec({:?})", &self));
        match self {
            Expr::Parenthesized(inner) => inner.exec(vars, funcs),
            Expr::FuncCall(e) => {
                let func = funcs.get(&*e.name).ok_or_else(|| ExecError::Other(format!("Undefined function '{}'", &e.name)))?;
                let args: Vec<DynArray> = e.args.iter().map(|e| e.exec(vars, funcs)).try_collect()?;
                //log::log(&format!("FuncCall::exec(fn: {:?}, args: {:?})", e.name, &args));
                func.call(&args)
            },
            Expr::Binary(e) => e.exec(vars, funcs),
            Expr::Unary(e) => e.exec(vars, funcs),
            Expr::Variable(v) => {
                vars.get(&v.0).map(|v| v.clone())
                    .ok_or_else(|| ExecError::Other(format!("Undefined variable '{}'", &v.0)))
            }
            Expr::Const(_name, v) => { Ok(DynArray::from_val(*v)) },
            Expr::Literal(LiteralExpr::Complex(v)) => { Ok(DynArray::from_val(*v)) },
            Expr::Literal(LiteralExpr::Float(v)) => { Ok(DynArray::from_val(*v)) },
            Expr::Literal(LiteralExpr::Int(v)) => { Ok(DynArray::from_val(*v)) },
            Expr::Literal(LiteralExpr::Array(v)) => { Ok(v.as_ref().clone()) },
        }
    }
}

impl BinaryExpr {
    pub fn exec(&self, vars: &HashMap<Rc<str>, DynArray>, funcs: &FuncMap) -> Result<DynArray, ExecError> {
        let lhs = self.lhs.exec(vars, funcs)?;
        let rhs = self.rhs.exec(vars, funcs)?;
        //log::log(&format!("BinaryExpr::exec(lhs: {:?}, op: {:?}, rhs: {:?})", &lhs, &self.op, &rhs));
        Ok(match self.op {
            BinaryOp::Add => { lhs + rhs },
            BinaryOp::Div => { lhs / rhs },
            BinaryOp::Mul => {
                match lhs.try_mul(rhs) {
                    Ok(val) => {
                        //log::log(&format!("returning: {:?}", &val));
                        val
                    }
                    Err(e) => {
                        //log::log(&format!("Error in mul: {}", e));
                        panic!("{}", e);
                    }
                }
            },
            BinaryOp::Sub => { lhs - rhs },

            BinaryOp::Rem => { lhs % rhs },
            BinaryOp::MatMul => { lhs.mat_mul(rhs) },
            BinaryOp::Pow => { lhs.pow(rhs) },

            BinaryOp::Eq => { lhs.equals(rhs) },
            BinaryOp::NotEq => { lhs.not_equals(rhs) },
            BinaryOp::GreaterThan => { lhs.greater(rhs) },
            BinaryOp::LessThan => { lhs.less(rhs) },
            BinaryOp::GreaterThanEq => { lhs.greater_equal(rhs) },
            BinaryOp::LessThanEq => { lhs.less_equal(rhs) },

            BinaryOp::BitAnd => { lhs & rhs },
            BinaryOp::BitOr => { lhs | rhs },
            BinaryOp::BitXor => { lhs ^ rhs },
        })
    }
}

impl UnaryExpr {
    pub fn exec(&self, vars: &HashMap<Rc<str>, DynArray>, funcs: &FuncMap) -> Result<DynArray, ExecError> {
        let inner = self.inner.exec(vars, funcs)?;
        //log::log(&format!("UnaryExpr::exec(op: {:?}, inner: {:?})", &self.op, &inner));
        Ok(match self.op {
            UnaryOp::Neg => { -inner },
            UnaryOp::Pos => { inner },
            UnaryOp::Not => { !inner },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum ParseError {
    InvalidLiteral(String),
    Other(String),
    UnexpectedEof(String),
    #[default]
    Unknown,
}

impl From<ParseIntError> for ParseError {
    fn from(value: ParseIntError) -> Self {
        ParseError::InvalidLiteral(format!("Invalid int literal: {}", value))
    }
}

impl From<ParseFloatError> for ParseError {
    fn from(value: ParseFloatError) -> Self {
        ParseError::InvalidLiteral(format!("Invalid float literal: {}", value))
    }
}

pub fn parse_with_literals<'a, I1: IntoIterator<Item = &'a str>, I2: IntoIterator<Item = Token>>(strings: I1, literals: I2) -> Result<Expr, ParseError> {
    // this is a big hack. don't think about it too hard
    let iter1 = strings.into_iter().map(|s| Ok(logos::Lexer::new(s)));
    let iter2 = literals.into_iter().map(|lit| Err(Ok(lit)));
    let mut lexer = Lexer::from(iter1.interleave(iter2).flatten_ok().map(|v| match v {
        Ok(v) => v,
        Err(v) => v,
    }));

    //return Err(ParseError::Other(format!("{:?}", lexer.inner.map(|v| v.unwrap()).collect_vec())));

    let expr = parse_expr(&mut lexer)?;
    // check we've parsed everything
    if let Some(token) = lexer.peek()? {
        Err(ParseError::Other(format!("Unexpected token {:?}", token)))
    } else { Ok(expr) }
}

pub fn parse<T: AsRef<str>>(input: T) -> Result<Expr, ParseError> {
    let mut lexer = Token::lexer(input.as_ref()).into();
    let expr = parse_expr(&mut lexer)?;
    // check we've parsed everything
    if let Some(token) = lexer.peek()? {
        Err(ParseError::Other(format!("Unexpected token {:?}", token)))
    } else { Ok(expr) }
}

fn parse_expr<I: Iterator<Item = Result<Token, ParseError>>>(lexer: &mut Lexer<I>) -> Result<Expr, ParseError> {
    let lhs = parse_unary(lexer)?;
    parse_binary(lexer, lhs, None)
}

fn parse_primary<I: Iterator<Item = Result<Token, ParseError>>>(lexer: &mut Lexer<I>) -> Result<Expr, ParseError> {
    match lexer.peek()? {
        Some(Token::OpenParen) => {
            lexer.next();
            let result = parse_expr(lexer)?;
            match lexer.peek()? {
                Some(Token::CloseParen) => lexer.next(),
                Some(token) => return Err(ParseError::Other(format!("Unexpected token {:?}. Expected close parenthesis", token))),
                None => return Err(ParseError::UnexpectedEof("Unexpected EOF before close parenthesis".to_owned())),
            }
            return Ok(Expr::Parenthesized(result.into()));
        }
        Some(Token::FuncOpen(name)) => {
            lexer.next();
            return parse_func(lexer, name);
        }
        _ => (),
    }

    match lexer.as_terminal()? {
        Some(s) => { lexer.next(); Ok(s) },
        None => match lexer.peek()? {
            Some(token) => Err(ParseError::Other(format!("Unexpected token {:?}. Expected a literal or variable", token))),
            None => Err(ParseError::UnexpectedEof("Unexpected EOF. Expected primary expression".to_owned())),
        }
    }
}

fn parse_func<I: Iterator<Item = Result<Token, ParseError>>>(lexer: &mut Lexer<I>, name: Rc<str>) -> Result<Expr, ParseError> {
    let mut args = Vec::new();
    loop {
        args.push(parse_expr(lexer)?);
        return match lexer.peek()? {
            Some(Token::CloseParen) => { lexer.next(); Ok(Expr::FuncCall(FuncExpr { name, args })) },
            Some(Token::Comma) => { lexer.next(); continue },
            Some(token) => Err(ParseError::Other(format!("Unexpected token {:?}. Expected a comma or close parenthesis", token))),
            None => Err(ParseError::UnexpectedEof("Unexpected EOF before close parenthesis".to_owned())),
        }
    }
}

fn parse_unary<I: Iterator<Item = Result<Token, ParseError>>>(lexer: &mut Lexer<I>) -> Result<Expr, ParseError> {
    if let Some(op) = lexer.as_unary_op()? {
        lexer.next();
        let inner = parse_unary(lexer)?.into();

        // fuse +/- operators with literals
        return Ok(match (op, inner) {
            (UnaryOp::Pos, Expr::Literal(LiteralExpr::Int(v))) => Expr::Literal(LiteralExpr::Int(v)),
            (UnaryOp::Pos, Expr::Literal(LiteralExpr::Float(v))) => Expr::Literal(LiteralExpr::Float(v)),
            (UnaryOp::Pos, Expr::Literal(LiteralExpr::Complex(v))) => Expr::Literal(LiteralExpr::Complex(v)),
            (UnaryOp::Neg, Expr::Literal(LiteralExpr::Int(v))) => Expr::Literal(LiteralExpr::Int(-v)),
            (UnaryOp::Neg, Expr::Literal(LiteralExpr::Float(v))) => Expr::Literal(LiteralExpr::Float(-v)),
            (UnaryOp::Neg, Expr::Literal(LiteralExpr::Complex(v))) => Expr::Literal(LiteralExpr::Complex(-v)),
            (op, inner) => Expr::Unary(UnaryExpr { op, inner: inner.into() }),
        })
    }

    parse_primary(lexer)
}

fn parse_binary<I: Iterator<Item = Result<Token, ParseError>>>(lexer: &mut Lexer<I>, mut lhs: Expr, outer_op: Option<&BinaryOp>) -> Result<Expr, ParseError> {
    // shunting yard parser
    loop {
        let op = match lexer.as_binary_op()? {
            Some(s) => s,
            None => break,
        };

        if let Some(outer_op) = outer_op {
            if !op.precedes(outer_op) {
                // this op is of lower precedence, it needs to be parsed at a higher level
                break;
            }
        }

        lexer.next();
        let mut rhs = parse_unary(lexer)?;

        if let Some(inner_op) = lexer.as_binary_op()? {
            if inner_op.precedes(&op) {
                // rhs is actually lhs of an inner_op expression
                // recurse to collect
                rhs = parse_binary(lexer, rhs, Some(&op))?;
            }
        }

        lhs = Expr::Binary(BinaryExpr { lhs: lhs.into(), op, rhs: rhs.into() })
    }
    Ok(lhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parse() {
        assert_eq!(
            parse("12_3e5"),
            Ok(Expr::Literal(LiteralExpr::Float(123e5))),
        );
    }

    #[test]
    fn test_binary_parse() {
        assert_eq!(
            parse("1 + a * +3"),
            Ok(Expr::Binary(BinaryExpr {
                lhs: Expr::Literal(LiteralExpr::Int(1)).into(),
                op: BinaryOp::Add,
                rhs: Expr::Binary(BinaryExpr {
                    lhs: Expr::Variable(VariableExpr(Rc::from("a"))).into(),
                    op: BinaryOp::Mul,
                    rhs: Expr::Literal(LiteralExpr::Int(3)).into(),
                }).into(),
            }))
        );

        assert_eq!(
            parse("1*2*3 +"),
            Err(ParseError::UnexpectedEof("Unexpected EOF. Expected primary expression".to_owned())),
        );
    }

    #[test]
    fn test_binary_parse_simple() {
        assert_eq!(
            parse("1+2+3"),
            Ok(Expr::Binary(BinaryExpr {
                lhs: Expr::Binary(BinaryExpr {
                    lhs: Expr::Literal(LiteralExpr::Int(1)).into(),
                    op: BinaryOp::Add,
                    rhs: Expr::Literal(LiteralExpr::Int(2)).into(),
                }).into(),
                op: BinaryOp::Add,
                rhs: Expr::Literal(LiteralExpr::Int(3)).into(),
            }))
        );
    }

    #[test]
    fn test_unary_parse() {
        assert_eq!(
            parse("1 + +-+a"),
            Ok(Expr::Binary(BinaryExpr {
                lhs: Expr::Literal(LiteralExpr::Int(1)).into(),
                op: BinaryOp::Add,
                rhs: Expr::Unary(UnaryExpr {
                    op: UnaryOp::Pos,
                    inner: Expr::Unary(UnaryExpr {
                        op: UnaryOp::Neg,
                        inner: Expr::Unary(UnaryExpr {
                            op: UnaryOp::Pos,
                            inner: Expr::Variable(VariableExpr(Rc::from("a"))).into()
                        }).into(),
                    }).into(),
                }).into(),
            }))
        )
    }

    #[test]
    fn test_func_parse() {
        assert_eq!(
            parse("abs(1 + 2, 4 + 5, c)"),
            Ok(Expr::FuncCall(FuncExpr {
                name: Rc::from("abs"),
                args: vec![
                    Expr::Binary(BinaryExpr {
                        lhs: Expr::Literal(LiteralExpr::Int(1)).into(),
                        op: BinaryOp::Add,
                        rhs: Expr::Literal(LiteralExpr::Int(2)).into(),
                    }),
                    Expr::Binary(BinaryExpr {
                        lhs: Expr::Literal(LiteralExpr::Int(4)).into(),
                        op: BinaryOp::Add,
                        rhs: Expr::Literal(LiteralExpr::Int(5)).into(),
                    }),
                    Expr::Variable(VariableExpr(Rc::from("c"))),
                ]
            }))
        )
    }

    #[test]
    fn test_complex_literals() {
        assert_eq!(
            parse("5.j"),
            Ok(Expr::Literal(LiteralExpr::Complex(Complex::new(0.0, 5.0)))),
        );

        assert_eq!(
            parse("1j"),
            Ok(Expr::Literal(LiteralExpr::Complex(Complex::new(0.0, 1.0)))),
        );

        assert_eq!(
            parse("1.2j+1.j+1j"),
            Ok(Expr::Binary(BinaryExpr {
                lhs: Expr::Binary(BinaryExpr {
                    lhs: Expr::Literal(LiteralExpr::Complex(Complex::new(0.0, 1.2))).into(),
                    op: BinaryOp::Add,
                    rhs: Expr::Literal(LiteralExpr::Complex(Complex::new(0.0, 1.0))).into(),
                }).into(),
                op: BinaryOp::Add,
                rhs: Expr::Literal(LiteralExpr::Complex(Complex::new(0.0, 1.0))).into(),
            }))
        );
    }
}