#![feature(new_uninit, type_alias_impl_trait, lazy_cell)]
#![feature(iterator_try_collect)]
#![feature(convert_float_to_int, float_minimum_maximum)]

pub mod bool;
pub mod dtype;
mod cast;
//pub mod typedarray;
pub mod error;
pub mod array;
pub mod fft;
pub mod reductions;
pub mod colors;
pub(crate) mod util;