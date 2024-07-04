use core::num::{ParseFloatError, ParseIntError};
use std::str::FromStr;
use std::rc::Rc;
use std::collections::HashMap;

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
    #[token("^")]
    Caret,
    #[regex(r"[+-]?(?&dec)", parse_int)]
    IntLit(i64),
    // inf, nan
    #[regex(r"[+-]?(inf|nan)", parse_float, ignore(ascii_case))]
    // 5., 5.e3
    #[regex(r"[+-]?(?&dec)\.(?&dec)?(?&exp)?", parse_float)]
    // .5, .5e-3
    #[regex(r"[+-]?\.(?&dec)(?&exp)?", parse_float)]
    // 1e5
    #[regex(r"[+-]?(?&dec)(?&exp)", parse_float)]
    FloatLit(f64),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9]*", |lexer| Rc::from(lexer.slice()))]
    Ident(Rc<str>),
    ArrayLit(DynArray),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9]*\(", |lexer| Rc::from(lexer.slice().strip_suffix('(').unwrap()))]
    FuncOpen(Rc<str>),
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
            Token::Caret => Some(BinaryOp::Pow),
            _ => None,
        }))
    }

    pub fn as_unary_op(&mut self) -> Result<Option<UnaryOp>, ParseError> {
        Ok(self.peek()?.and_then(|token| match token {
            Token::Plus => Some(UnaryOp::Pos),
            Token::Minus => Some(UnaryOp::Neg),
            _ => None,
        }))
    }

    pub fn as_terminal(&mut self) -> Result<Option<Expr>, ParseError> {
        Ok(match self.peek()? {
            Some(Token::IntLit(value)) => Some(Expr::Literal(LiteralExpr::Int(value))),
            Some(Token::FloatLit(value)) => Some(Expr::Literal(LiteralExpr::Float(value))),
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
}

impl BinaryOp {
    pub fn precedence(&self) -> i64 {
        match self {
            BinaryOp::Rem => 4,
            BinaryOp::Add | BinaryOp::Sub => 5,
            BinaryOp::Mul | BinaryOp::Div => 6,
            BinaryOp::Pow => 7,
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
    Float(f64),
    Int(i64),
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

impl Expr {
    pub fn exec(&self, vars: &HashMap<Rc<str>, DynArray>) -> Result<DynArray, ExecError> {
        match self {
            Expr::Parenthesized(inner) => inner.exec(vars),
            Expr::FuncCall(_e) => unimplemented!(),
            Expr::Binary(e) => e.exec(vars),
            Expr::Unary(e) => e.exec(vars),
            Expr::Variable(v) => {
                vars.get(&v.0).map(|v| v.clone())
                    .ok_or_else(|| ExecError::Other(format!("Undefined variable '{}'", &v.0)))
            }
            Expr::Literal(LiteralExpr::Float(v)) => { Ok(DynArray::full(vec![], *v)) },
            Expr::Literal(LiteralExpr::Int(v)) => { Ok(DynArray::full(vec![], *v)) },
            Expr::Literal(LiteralExpr::Array(v)) => { Ok(v.as_ref().clone()) },
        }
    }
}

impl BinaryExpr {
    pub fn exec(&self, vars: &HashMap<Rc<str>, DynArray>) -> Result<DynArray, ExecError> {
        let lhs = self.lhs.exec(vars)?;
        let rhs = self.rhs.exec(vars)?;
        Ok(match self.op {
            BinaryOp::Add => { lhs + rhs },
            BinaryOp::Div => { lhs / rhs },
            BinaryOp::Mul => { lhs * rhs },
            BinaryOp::Sub => { lhs - rhs },
            BinaryOp::Rem => { lhs % rhs },
            BinaryOp::Pow => { lhs.pow(rhs) },
        })
    }
}

impl UnaryExpr {
    pub fn exec(&self, vars: &HashMap<Rc<str>, DynArray>) -> Result<DynArray, ExecError> {
        let inner = self.inner.exec(vars)?;
        Ok(match self.op {
            UnaryOp::Neg => { -inner },
            UnaryOp::Pos => { inner },
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
        return Ok(Expr::Unary(UnaryExpr { op, inner }));
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
}