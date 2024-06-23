use std::any::Any;
use std::ops;
use std::borrow::{Borrow, Cow};
use std::fmt;

use bytemuck::Pod;
use num_complex::{Complex, ComplexFloat};
use ndarray::{Array, ArrayD, Dimension, IxDyn, ShapeBuilder, ShapeError, ErrorKind, Zip};

use arraylib_macro::{type_dispatch, forward_val_to_ref};
use crate::dtype::{DataType, PhysicalType, Bool, promote_types};
use crate::cast::Cast;
use crate::error::ArrayError;

#[derive(Debug)]
pub struct DynArray {
    dtype: DataType,
    shape: Vec<usize>,
    inner: Box<dyn Any>,
}

impl Clone for DynArray {
    fn clone(&self) -> Self {
        let s = self;

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| s.clone().into()
        )
    }
}

impl DynArray {
    pub fn dtype(&self) -> DataType { self.dtype }

    pub fn shape(&self) -> Vec<usize> { self.shape.clone() }

    pub fn from_typed<T: PhysicalType + Pod>(arr: ArrayD<T>) -> Self {
        Self {
            dtype: T::DATATYPE,
            shape: arr.shape().into(),
            inner: Box::new(arr) as Box<dyn Any>,
        }
    }

    pub fn downcast<T: PhysicalType + Pod>(self) -> Option<ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(*self.inner.downcast().unwrap())
    }

    pub fn downcast_ref<T: PhysicalType + Pod>(&self) -> Option<&ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(self.inner.downcast_ref().unwrap())
    }

    pub fn downcast_mut<T: PhysicalType + Pod>(&mut self) -> Option<&mut ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(self.inner.downcast_mut().unwrap())
    }

    pub fn zeros<Sh: ShapeBuilder<Dim = IxDyn>>(shape: Sh, dtype: DataType) -> Self {
        match dtype {
            DataType::Boolean => ArrayD::<Bool>::zeros(shape).into(),
            DataType::UInt8 => ArrayD::<u8>::zeros(shape).into(),
            DataType::UInt16 => ArrayD::<u16>::zeros(shape).into(),
            DataType::UInt32 => ArrayD::<u32>::zeros(shape).into(),
            DataType::UInt64 => ArrayD::<u64>::zeros(shape).into(),
            DataType::Int8 => ArrayD::<i8>::zeros(shape).into(),
            DataType::Int16 => ArrayD::<i16>::zeros(shape).into(),
            DataType::Int32 => ArrayD::<i32>::zeros(shape).into(),
            DataType::Int64 => ArrayD::<i64>::zeros(shape).into(),
            DataType::Float32 => ArrayD::<f32>::zeros(shape).into(),
            DataType::Float64 => ArrayD::<f64>::zeros(shape).into(),
            DataType::Complex64 => ArrayD::<Complex<f32>>::zeros(shape).into(),
            DataType::Complex128 => ArrayD::<Complex<f64>>::zeros(shape).into(),
        }
    }

    pub fn ones<Sh: ShapeBuilder<Dim = IxDyn>>(shape: Sh, dtype: DataType) -> Self {
        match dtype {
            DataType::Boolean => ArrayD::<Bool>::ones(shape).into(),
            DataType::UInt8 => ArrayD::<u8>::ones(shape).into(),
            DataType::UInt16 => ArrayD::<u16>::ones(shape).into(),
            DataType::UInt32 => ArrayD::<u32>::ones(shape).into(),
            DataType::UInt64 => ArrayD::<u64>::ones(shape).into(),
            DataType::Int8 => ArrayD::<i8>::ones(shape).into(),
            DataType::Int16 => ArrayD::<i16>::ones(shape).into(),
            DataType::Int32 => ArrayD::<i32>::ones(shape).into(),
            DataType::Int64 => ArrayD::<i64>::ones(shape).into(),
            DataType::Float32 => ArrayD::<f32>::ones(shape).into(),
            DataType::Float64 => ArrayD::<f64>::ones(shape).into(),
            DataType::Complex64 => ArrayD::<Complex<f32>>::ones(shape).into(),
            DataType::Complex128 => ArrayD::<Complex<f64>>::ones(shape).into(),
        }
    }

    pub fn full<T: PhysicalType + Pod, Sh: ShapeBuilder<Dim = IxDyn>>(shape: Sh, value: T) -> Self {
        ArrayD::<T>::from_elem(shape, value).into()
    }

    pub fn broadcast_with(&self, other: &DynArray) -> Result<(DynArray, DynArray), ShapeError> {
        let (lhs, rhs) = (self, other);
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| {
                let broadcast_shape: IxDyn = co_broadcast(&lhs.raw_dim().into_dyn(), &rhs.raw_dim().into_dyn())?;
                Ok((lhs.broadcast(broadcast_shape.clone()).unwrap().mapv(|v| v).into(), rhs.broadcast(broadcast_shape).unwrap().mapv(|v| v).into()))
            }
        )
    }
}

pub(crate) fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in out.slice_mut().into_iter().zip(shape1.slice()) {
        *out = *s;
    }
    for (out, s2) in (&mut out.slice_mut()[k..]).into_iter().zip(shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

impl<T: PhysicalType + Pod, D: Dimension> From<Array<T, D>> for DynArray {
    fn from(value: Array<T, D>) -> Self { Self::from_typed(value.into_dyn()) }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::Add<T> for &'a DynArray {
    type Output = DynArray;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            // use wrapping arithmetic for integral types
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1.wrapping_add(e2)).into() },
            (Bool, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 + e2).into() },
        )
    }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::Sub<T> for &'a DynArray {
    type Output = DynArray;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            // use wrapping arithmetic for integral types
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1.wrapping_sub(e2)).into() },
            (Bool, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 - e2).into() },
        )
    }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::Mul<T> for &'a DynArray {
    type Output = DynArray;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1.wrapping_mul(e2)).into() },
            (Bool, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 * e2).into() },
        )
    }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::Div<T> for &'a DynArray {
    type Output = DynArray;

    fn div(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            // saturating div, ensuring i32::MIN / -1i32 == i32::MAX
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1.saturating_div(e2)).into() },
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 / e2).into() },
        )
    }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::BitAnd<T> for &'a DynArray {
    type Output = DynArray;

    fn bitand(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 & e2).into() },
        )
    }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::Rem<T> for &'a DynArray {
    type Output = DynArray;

    fn rem(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();
        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 % e2).into() }
        )
    }
}

impl<T: Borrow<DynArray>> ops::AddAssign<T> for DynArray {
    fn add_assign(&mut self, other: T) {
        let (rhs, lhs) = (other.borrow().cast(self.dtype), self);

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref mut lhs, ref rhs| { lhs.zip_mut_with(rhs, |e1, e2| *e1 += *e2) }
        );
    }
}

impl<T: Borrow<DynArray>> ops::SubAssign<T> for DynArray {
    fn sub_assign(&mut self, other: T) {
        let (rhs, lhs) = (other.borrow().cast(self.dtype), self);

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref mut lhs, ref rhs| { lhs.zip_mut_with(rhs, |e1, e2| *e1 -= *e2) }
        );
    }
}

impl ops::Neg for DynArray {
    type Output = DynArray;

    fn neg(self) -> Self::Output {
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64),
            |ref s| {s.mapv(|v| !v).into() },
            (i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|v| -v).into() },
        )
    }
}

// comparison operators
impl PartialEq for DynArray {
    fn eq(&self, other: &DynArray) -> bool {
        let (lhs, rhs) = (self, other);
        return lhs.dtype == rhs.dtype && (
            type_dispatch!(
                (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
                |ref lhs, ref rhs| { lhs == rhs }
            )
        )
    }
}
impl Eq for DynArray {}


macro_rules! cast_to_impl {
    ($arr:expr, $dtype:expr, $( ($ty:path, $fn:ident) ),* ) => {
        match $dtype  {
            $(
                <$ty as PhysicalType>::DATATYPE => { let f = T::$fn()?; Some($arr.mapv(f).into()) }
            ),*,
        }
    };
}

#[inline]
fn cast_to<T: Cast + PhysicalType>(arr: &ArrayD<T>, dtype: DataType) -> Option<DynArray> {
    cast_to_impl!(arr, dtype,
        (Bool, cast_bool),
        (u8, cast_uint8), (u16, cast_uint16), (u32, cast_uint32), (u64, cast_uint64),
        (i8, cast_int8), (i16, cast_int16), (i32, cast_int32), (i64, cast_int64),
        (f32, cast_float32), (f64, cast_float64),
        (Complex<f32>, cast_complex64), (Complex<f64>, cast_complex128)
    )
}

impl DynArray {
    pub fn cast<'a>(&'a self, dtype: DataType) -> Cow<'a, DynArray> {
        let init_dtype = self.dtype();
        if init_dtype == dtype {
            return Cow::Borrowed(self);
        }

        let s = self;
        match type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| cast_to(s, dtype)
        ) {
            Some(arr) => Cow::Owned(arr),
            None => std::panic::panic_any(ArrayError::type_err(format!("Unable to cast dtype {} to {}", init_dtype, dtype))),
        }
    }

    pub fn abs(self) -> DynArray {
        let s = self;

        type_dispatch!(
            (u8, u16, u32, u64),
            |ref s| { s.clone().into() },
            // saturating abs, ensuring abs(i32::MIN) == i32::MAX
            (i8, i16, i32, i64),
            |ref s| { s.mapv(|e| e.saturating_abs()).into() },
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.abs()).into() },
        )
    }

    pub fn equals<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l == r)).into() }
        )
    }

    pub fn not_equals<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l != r)).into() }
        )
    }

    /*
    fn calc_max_elem_width_display(&self) -> usize {
        let mut buf = String::with_capacity(64);
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.fold(0usize, |accum, v| {
                buf.clear();
                buf.write_fmt(format_args!("{}", v)).unwrap();
                accum.max(buf.len())
            }) }
        )
    }
    */
}

impl fmt::Display for DynArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { fmt::Display::fmt(s, f) }
        )
    }
}

impl fmt::Binary for DynArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { fmt::Binary::fmt(s, f) }
        )
    }
}

impl fmt::LowerExp for DynArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { fmt::LowerExp::fmt(s, f) }
        )
    }
}

impl fmt::UpperExp for DynArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { fmt::UpperExp::fmt(s, f) }
        )
    }
}


impl fmt::LowerHex for DynArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { fmt::LowerHex::fmt(s, f) }
        )
    }
}