use num::complex::Complex;
use std::fmt::{Display, Formatter};
use std::mem::{size_of, MaybeUninit};

use bytemuck::AnyBitPattern;

pub use crate::bool::Bool;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataType {
    Boolean,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

pub trait PhysicalType: AnyBitPattern {
    const DATATYPE: DataType;
    type BytesType;
}

impl<T: PhysicalType + AnyBitPattern> PhysicalType for MaybeUninit<T> {
    const DATATYPE: DataType = T::DATATYPE;
    type BytesType = T::BytesType;
}

impl PhysicalType for Bool { const DATATYPE: DataType = DataType::Boolean; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for u8 { const DATATYPE: DataType = DataType::UInt8; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for u16 { const DATATYPE: DataType = DataType::UInt16; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for u32 { const DATATYPE: DataType = DataType::UInt32; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for u64 { const DATATYPE: DataType = DataType::UInt64; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for i8 { const DATATYPE: DataType = DataType::Int8; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for i16 { const DATATYPE: DataType = DataType::Int16; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for i32 { const DATATYPE: DataType = DataType::Int32; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for i64 { const DATATYPE: DataType = DataType::Int64; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for f32 { const DATATYPE: DataType = DataType::Float32; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for f64 { const DATATYPE: DataType = DataType::Float64; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for Complex<f32> { const DATATYPE: DataType = DataType::Complex64; type BytesType = [u8; size_of::<Self>()]; }
impl PhysicalType for Complex<f64> { const DATATYPE: DataType = DataType::Complex128; type BytesType = [u8; size_of::<Self>()]; }

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DataType::Boolean => "bool",
            DataType::UInt8 => "u8",
            DataType::UInt16 => "u16",
            DataType::UInt32 => "u32",
            DataType::UInt64 => "u64",
            DataType::Int8 => "i8",
            DataType::Int16 => "i16",
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::Complex64 => "c64",
            DataType::Complex128 => "c128",
        })
    }
}