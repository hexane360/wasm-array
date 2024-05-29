use std::borrow::{Borrow, Cow};
use std::fmt;

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ArrayError {
    BroadcastError(BroadcastError),
    TypeError(TypeError),
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayError::BroadcastError(e) => e.fmt(f),
            ArrayError::TypeError(e) => e.fmt(f),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastError(Cow<'static, str>);

impl fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0.borrow()) }
}

#[derive(Debug, Clone)]
pub struct TypeError(Cow<'static, str>);

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0.borrow()) }
}

impl ArrayError {
    pub fn broadcast_err<I: Into<Cow<'static, str>>>(msg: I) -> Self {
        ArrayError::BroadcastError(BroadcastError(msg.into()))
    }

    pub fn type_err<I: Into<Cow<'static, str>>>(msg: I) -> Self {
        ArrayError::TypeError(TypeError(msg.into()))
    }
}