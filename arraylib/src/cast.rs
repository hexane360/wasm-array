use std::convert::FloatToInt;

use num::{Float, PrimInt, Zero, Signed};

use crate::dtype::{Bool, Complex};

// this is a horrible hack
const fn cast_strategy_none<T, U>() -> Option<fn(T) -> U> {
    match false {
        false => None,
        true => Some(|_| unreachable!()),
    }
}

const fn cast_strategy_into<T, U>() -> Option<fn(T) -> U>
    where T: Into<U>
{
    Some(<T as Into<U>>::into)
}

const fn cast_strategy_bool_to_num<I>() -> Option<fn(Bool) -> I>
where I: From<bool>
{ Some(|s: Bool| { s.canonicalize().into() }) }

const fn cast_strategy_float_to_int<F, I>() -> Option<fn(F) -> I>
where F: FloatToInt<I>, F: Float, I: PrimInt + Signed,
{
    Some(|s: F| {
        if s.is_finite() {
            // TODO bounds check here
            unsafe { s.to_int_unchecked() }
        } else if s.is_nan() {
            I::zero() * (if s.is_sign_positive() { I::one() } else { -I::one() })
        } else {
            if s.is_sign_positive() { I::max_value() } else { I::min_value() }
        }
    })
}

const fn cast_strategy_float_to_uint<F, I>() -> Option<fn(F) -> I>
where F: FloatToInt<I>, F: Float, I: PrimInt,
{
    Some(|s: F| {
        if s.is_finite() {
            unsafe { if s >= F::zero() { s.to_int_unchecked() } else { I::max_value() - s.to_int_unchecked() } }
        } else if s.is_nan() {
            I::zero() * (if s.is_sign_positive() { I::one() } else { I::max_value() - I::one() })
        } else {
            if s.is_sign_positive() { I::max_value() } else { I::min_value() }
        }
    })
}

const fn cast_strategy_to_bool<T>() -> Option<fn(T) -> Bool>
where T: Zero + PartialEq { Some(|s| Bool::from(s != T::zero())) }

// `as` casting has most of the properties we want.
// TODO: replace this with something more explicit,
// recording invalid values
macro_rules! cast_strategy_cast {
    ( $T:ty, $U:ty ) => {
        Some(|s: $T| s as $U)
    };
}

pub(crate) trait Cast: Sized {
    fn cast_bool() -> Option<fn(Self) -> Bool>;
    fn cast_uint8() -> Option<fn(Self) -> u8>;
    fn cast_uint16() -> Option<fn(Self) -> u16>;
    fn cast_uint32() -> Option<fn(Self) -> u32>;
    fn cast_uint64() -> Option<fn(Self) -> u64>;
    fn cast_int8() -> Option<fn(Self) -> i8>;
    fn cast_int16() -> Option<fn(Self) -> i16>;
    fn cast_int32() -> Option<fn(Self) -> i32>;
    fn cast_int64() -> Option<fn(Self) -> i64>;
    fn cast_float32() -> Option<fn(Self) -> f32>;
    fn cast_float64() -> Option<fn(Self) -> f64>;
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>>;
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>>;
}

impl Cast for Bool {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_bool_to_num() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_bool_to_num() }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_bool_to_num() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_bool_to_num() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_bool_to_num() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_bool_to_num() }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_bool_to_num() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_bool_to_num() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_bool_to_num() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_bool_to_num() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_bool_to_num() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s: Bool| f32::from(s.canonicalize()).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s: Bool| f64::from(s.canonicalize()).into()) }
}

impl Cast for u8 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_into() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_into() }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_into() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_into() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(u8, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_into() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for u16 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(u16, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_into() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_into() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(u16, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_cast!(u16, i16) }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for u32 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(u32, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_cast!(u32, u16) }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_into() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(u32, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_cast!(u32, i16) }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_cast!(u32, i32) }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_cast!(u32, f32) }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for u64 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(u64, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_cast!(u64, u16) }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_cast!(u64, u32) }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(u64, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_cast!(u64, i16) }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_cast!(u64, i32) }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_cast!(u64, i64) }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_cast!(u64, f32) }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_cast!(u64, f64) }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| (s as f64).into()) }
}

impl Cast for i8 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(i8, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_cast!(i8, u16) }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_cast!(i8, u32) }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_cast!(i8, u64) }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_into() }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_into() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for i16 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(i16, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_cast!(i16, u16) }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_cast!(i16, u32) }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_cast!(i16, u64) }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(i16, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_into() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for i32 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(i32, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_cast!(i32, u16) }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_cast!(i32, u32) }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_cast!(i32, u64) }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(i32, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_cast!(i32, i16) }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_cast!(i32, f32) }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for i64 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_cast!(i64, u8) }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_cast!(i64, u16) }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_cast!(i64, u32) }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_cast!(i64, u64) }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_cast!(i64, i8) }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_cast!(i64, i16) }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_cast!(i64, i32) }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_cast!(i64, f32) }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_cast!(i64, f64) }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| (s as f64).into()) }
}

impl Cast for f32 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_float_to_uint() }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_float_to_uint() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_float_to_uint() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_float_to_uint() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_float_to_int() }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_float_to_int() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_float_to_int() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_float_to_int() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(f32::into) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s: f32| f64::from(s).into()) }
}

impl Cast for f64 {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_float_to_uint() }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_float_to_uint() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_float_to_uint() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_float_to_uint() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_float_to_int() }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_float_to_int() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_float_to_int() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_float_to_int() }
    fn cast_float32() -> Option<fn(Self) -> f32> { Some(|s| s as f32) }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|s| Complex::new(s as f32, 0.) ) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|s| Complex::new(s, 0.) ) }
}

impl Cast for Complex<f32> {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_none() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_none() }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_none() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_none() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_none() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_none() }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_none() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_none() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_none() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_none() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_none() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { cast_strategy_into() }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { Some(|v: Complex<f32>| Complex::new(v.re.into(), v.im.into())) }
}

impl Cast for Complex<f64> {
    fn cast_bool() -> Option<fn(Self) -> Bool> { cast_strategy_none() }
    fn cast_uint8() -> Option<fn(Self) -> u8> { cast_strategy_none() }
    fn cast_uint16() -> Option<fn(Self) -> u16> { cast_strategy_none() }
    fn cast_uint32() -> Option<fn(Self) -> u32> { cast_strategy_none() }
    fn cast_uint64() -> Option<fn(Self) -> u64> { cast_strategy_none() }
    fn cast_int8() -> Option<fn(Self) -> i8> { cast_strategy_none() }
    fn cast_int16() -> Option<fn(Self) -> i16> { cast_strategy_none() }
    fn cast_int32() -> Option<fn(Self) -> i32> { cast_strategy_none() }
    fn cast_int64() -> Option<fn(Self) -> i64> { cast_strategy_none() }
    fn cast_float32() -> Option<fn(Self) -> f32> { cast_strategy_none() }
    fn cast_float64() -> Option<fn(Self) -> f64> { cast_strategy_none() }
    fn cast_complex64() -> Option<fn(Self) -> Complex<f32>> { Some(|v: Complex<f64>| Complex::new(v.re as f32, v.im as f32)) }
    fn cast_complex128() -> Option<fn(Self) -> Complex<f64>> { cast_strategy_into() }
}

#[allow(dead_code)]
pub(crate) trait CastFrom: Sized {
    fn cast_from_bool() -> Option<fn(Bool) -> Self>;
    fn cast_from_uint8() -> Option<fn(u8) -> Self>;
    fn cast_from_uint16() -> Option<fn(u16) -> Self>;
    fn cast_from_uint32() -> Option<fn(u32) -> Self>;
    fn cast_from_uint64() -> Option<fn(u64) -> Self>;
    fn cast_from_int8() -> Option<fn(i8) -> Self>;
    fn cast_from_int16() -> Option<fn(i16) -> Self>;
    fn cast_from_int32() -> Option<fn(i32) -> Self>;
    fn cast_from_int64() -> Option<fn(i64) -> Self>;
    fn cast_from_float32() -> Option<fn(f32) -> Self>;
    fn cast_from_float64() -> Option<fn(f64) -> Self>;
    fn cast_from_complex64() -> Option<fn(Complex<f32>) -> Self>;
    fn cast_from_complex128() -> Option<fn(Complex<f64>) -> Self>;
}

macro_rules! impl_castfrom {
    ( $ty:ty, $fn:ident ) => {
        impl CastFrom for $ty {
            fn cast_from_bool() -> Option<fn(Bool) -> Self> { <Bool as Cast>::$fn() }
            fn cast_from_uint8() -> Option<fn(u8) -> Self> { <u8 as Cast>::$fn() }
            fn cast_from_uint16() -> Option<fn(u16) -> Self> { <u16 as Cast>::$fn() }
            fn cast_from_uint32() -> Option<fn(u32) -> Self> { <u32 as Cast>::$fn() }
            fn cast_from_uint64() -> Option<fn(u64) -> Self> { <u64 as Cast>::$fn() }
            fn cast_from_int8() -> Option<fn(i8) -> Self> { <i8 as Cast>::$fn() }
            fn cast_from_int16() -> Option<fn(i16) -> Self> { <i16 as Cast>::$fn() }
            fn cast_from_int32() -> Option<fn(i32) -> Self> { <i32 as Cast>::$fn() }
            fn cast_from_int64() -> Option<fn(i64) -> Self> { <i64 as Cast>::$fn() }
            fn cast_from_float32() -> Option<fn(f32) -> Self> { <f32 as Cast>::$fn() }
            fn cast_from_float64() -> Option<fn(f64) -> Self> { <f64 as Cast>::$fn() }
            fn cast_from_complex64() -> Option<fn(Complex<f32>) -> Self> { <Complex<f32> as Cast>::$fn() }
            fn cast_from_complex128() -> Option<fn(Complex<f64>) -> Self> { <Complex<f64> as Cast>::$fn() }
        }
    };
    ( $( ($ty:ty, $fn:ident) ),* ) => {
        $( impl_castfrom!($ty, $fn); )*
    }
}

impl_castfrom!(
    (Bool, cast_bool),
    (u8, cast_uint8),
    (u16, cast_uint16),
    (u32, cast_uint32),
    (u64, cast_uint64),
    (i8, cast_int8),
    (i16, cast_int16),
    (i32, cast_int32),
    (i64, cast_int64),
    (f32, cast_float32),
    (f64, cast_float64),
    (Complex<f32>, cast_complex64),
    (Complex<f64>, cast_complex128)
);