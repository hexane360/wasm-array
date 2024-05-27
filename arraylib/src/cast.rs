use std::convert::FloatToInt;

use num::{Float, PrimInt, Zero, Signed};

use crate::dtype::{Bool, Complex};

// this is a horrible hack
const fn cast_strategy_none<T, U>() -> Option<impl Fn(T) -> U> {
    match false {
        false => None,
        true => Some(|_| unreachable!()),
    }
}

const fn cast_strategy_into<T, U>() -> Option<impl Fn(T) -> U>
    where T: Into<U>
{
    Some(<T as Into<U>>::into)
}

const fn cast_strategy_bool_to_num<I>() -> Option<impl Fn(Bool) -> I>
where I: From<bool>
{ Some(|s: Bool| { s.canonicalize().into() }) }

const fn cast_strategy_float_to_int<F, I>() -> Option<impl Fn(F) -> I>
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

const fn cast_strategy_float_to_uint<F, I>() -> Option<impl Fn(F) -> I>
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

const fn cast_strategy_to_bool<T>() -> Option<impl Fn(T) -> Bool>
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
    fn cast_bool() -> Option<impl Fn(Self) -> Bool>;
    fn cast_uint8() -> Option<impl Fn(Self) -> u8>;
    fn cast_uint16() -> Option<impl Fn(Self) -> u16>;
    fn cast_uint32() -> Option<impl Fn(Self) -> u32>;
    fn cast_uint64() -> Option<impl Fn(Self) -> u64>;
    fn cast_int8() -> Option<impl Fn(Self) -> i8>;
    fn cast_int16() -> Option<impl Fn(Self) -> i16>;
    fn cast_int32() -> Option<impl Fn(Self) -> i32>;
    fn cast_int64() -> Option<impl Fn(Self) -> i64>;
    fn cast_float32() -> Option<impl Fn(Self) -> f32>;
    fn cast_float64() -> Option<impl Fn(Self) -> f64>;
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>>;
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>>;
}

impl Cast for Bool {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_bool_to_num() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_bool_to_num() }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_bool_to_num() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_bool_to_num() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_bool_to_num() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_bool_to_num() }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_bool_to_num() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_bool_to_num() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_bool_to_num() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_bool_to_num() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_bool_to_num() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s: Bool| f32::from(s.canonicalize()).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s: Bool| f64::from(s.canonicalize()).into()) }
}

impl Cast for u8 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_into() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_into() }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_into() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_into() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(u8, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_into() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for u16 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(u16, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_into() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_into() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(u16, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_cast!(u16, i16) }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for u32 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(u32, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_cast!(u32, u16) }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_into() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(u32, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_cast!(u32, i16) }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_cast!(u32, i32) }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_cast!(u32, f32) }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for u64 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(u64, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_cast!(u64, u16) }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_cast!(u64, u32) }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_into() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(u64, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_cast!(u64, i16) }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_cast!(u64, i32) }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_cast!(u64, i64) }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_cast!(u64, f32) }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_cast!(u64, f64) }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| (s as f64).into()) }
}

impl Cast for i8 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(i8, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_cast!(i8, u16) }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_cast!(i8, u32) }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_cast!(i8, u64) }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_into() }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_into() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for i16 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(i16, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_cast!(i16, u16) }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_cast!(i16, u32) }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_cast!(i16, u64) }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(i16, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_into() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| f32::from(s).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for i32 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(i32, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_cast!(i32, u16) }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_cast!(i32, u32) }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_cast!(i32, u64) }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(i32, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_cast!(i32, i16) }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_into() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_cast!(i32, f32) }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| f64::from(s).into()) }
}

impl Cast for i64 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_cast!(i64, u8) }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_cast!(i64, u16) }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_cast!(i64, u32) }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_cast!(i64, u64) }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_cast!(i64, i8) }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_cast!(i64, i16) }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_cast!(i64, i32) }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_into() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_cast!(i64, f32) }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_cast!(i64, f64) }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| (s as f32).into()) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| (s as f64).into()) }
}

impl Cast for f32 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_float_to_uint() }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_float_to_uint() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_float_to_uint() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_float_to_uint() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_float_to_int() }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_float_to_int() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_float_to_int() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_float_to_int() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_into() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(f32::into) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s: f32| f64::from(s).into()) }
}

impl Cast for f64 {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_to_bool() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_float_to_uint() }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_float_to_uint() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_float_to_uint() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_float_to_uint() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_float_to_int() }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_float_to_int() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_float_to_int() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_float_to_int() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { Some(|s| s as f32) }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_into() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|s| Complex::new(s as f32, 0.) ) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|s| Complex::new(s, 0.) ) }
}

impl Cast for Complex<f32> {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_none() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_none() }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_none() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_none() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_none() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_none() }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_none() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_none() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_none() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_none() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_none() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { cast_strategy_into() }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { Some(|v: Complex<f32>| Complex::new(v.re.into(), v.im.into())) }
}

impl Cast for Complex<f64> {
    fn cast_bool() -> Option<impl Fn(Self) -> Bool> { cast_strategy_none() }
    fn cast_uint8() -> Option<impl Fn(Self) -> u8> { cast_strategy_none() }
    fn cast_uint16() -> Option<impl Fn(Self) -> u16> { cast_strategy_none() }
    fn cast_uint32() -> Option<impl Fn(Self) -> u32> { cast_strategy_none() }
    fn cast_uint64() -> Option<impl Fn(Self) -> u64> { cast_strategy_none() }
    fn cast_int8() -> Option<impl Fn(Self) -> i8> { cast_strategy_none() }
    fn cast_int16() -> Option<impl Fn(Self) -> i16> { cast_strategy_none() }
    fn cast_int32() -> Option<impl Fn(Self) -> i32> { cast_strategy_none() }
    fn cast_int64() -> Option<impl Fn(Self) -> i64> { cast_strategy_none() }
    fn cast_float32() -> Option<impl Fn(Self) -> f32> { cast_strategy_none() }
    fn cast_float64() -> Option<impl Fn(Self) -> f64> { cast_strategy_none() }
    fn cast_complex64() -> Option<impl Fn(Self) -> Complex<f32>> { Some(|v: Complex<f64>| Complex::new(v.re as f32, v.im as f32)) }
    fn cast_complex128() -> Option<impl Fn(Self) -> Complex<f64>> { cast_strategy_into() }
}