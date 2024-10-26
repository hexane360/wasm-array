#![feature(type_alias_impl_trait)]
#![feature(iterator_try_collect)]
#![feature(convert_float_to_int, float_minimum_maximum)]

pub mod log;
pub mod bool;
pub mod dtype;
mod cast;
//pub mod typedarray;
pub mod error;
pub mod array;
pub mod fft;
pub mod reductions;
pub mod colors;
pub mod arraylike;
pub(crate) mod util;