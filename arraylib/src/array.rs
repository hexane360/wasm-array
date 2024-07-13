use std::any::Any;
use std::ops;
use std::borrow::{Borrow, Cow};
use std::fmt;
use std::panic::{RefUnwindSafe, UnwindSafe};

use bytemuck::Pod;
use itertools::Itertools;
use num::{Float, Zero, One, Integer};
use num_complex::{Complex, ComplexFloat};
use ndarray::{Array, Array1, Array2, ArrayD, Dimension, IxDyn, ShapeBuilder, ShapeError, ErrorKind, Zip, SliceInfoElem};

use arraylib_macro::{type_dispatch, forward_val_to_ref};
use crate::dtype::{DataType, DataTypeCategory, PhysicalType, Bool, promote_types};
use crate::cast::Cast;
use crate::error::ArrayError;
use crate::colors::{magma, apply_cmap_u8};
use crate::util::normalize_axis;

pub struct DynArray {
    dtype: DataType,
    shape: Vec<usize>,
    inner: Box<dyn Any + UnwindSafe + RefUnwindSafe>,
}

impl fmt::Debug for DynArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self;
        f.debug_struct("DynArray")
            .field("dtype", &s.dtype)
            .field("shape", &s.shape)
            .field("inner", &type_dispatch!(
                (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
                |ref s| s as &dyn fmt::Debug
            )).finish()
    }
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

    pub fn ndim(&self) -> usize { self.shape.len() }

    pub fn from_val<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe>(val: T) -> Self {
        Self::from_typed(ArrayD::from_elem(IxDyn(&[]), val))
    }

    pub fn from_typed<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe>(arr: ArrayD<T>) -> Self {
        Self {
            dtype: T::DATATYPE,
            shape: arr.shape().into(),
            inner: Box::new(arr) as Box<dyn Any + UnwindSafe + RefUnwindSafe>,
        }
    }

    pub fn downcast<T: PhysicalType + Pod>(self) -> Option<ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(*(self.inner as Box<dyn Any>).downcast().unwrap())
    }

    pub fn downcast_ref<T: PhysicalType + Pod>(&self) -> Option<&ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some((self.inner.as_ref() as &dyn Any).downcast_ref().unwrap())
    }

    pub fn downcast_mut<T: PhysicalType + Pod>(&mut self) -> Option<&mut ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some((self.inner.as_mut() as &mut dyn Any).downcast_mut().unwrap())
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

    pub fn from_buf(buf: Box<[u8]>, dtype: DataType, shape: Box<[usize]>, strides: Option<Box<[isize]>>) -> Result<Self, String> {
        //if strides.iter().any(|s| *s < 0) {
        //    return Err("Negative strides are unsupported".to_owned());
        //}
        // cast isize to usize, this should handle negative strides correctly
        let shape = match strides {
            Some(strides) => shape.strides(bytemuck::cast_slice(&strides)),
            // default to c order
            None => shape.into_shape().into(),
        };

        Ok(match dtype {
            DataType::UInt8 => ArrayD::<u8>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Boolean => ArrayD::<Bool>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::UInt16 => ArrayD::<u16>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::UInt32 => ArrayD::<u32>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::UInt64 => ArrayD::<u64>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Int8 => ArrayD::<i8>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Int16 => ArrayD::<i16>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Int32 => ArrayD::<i32>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Int64 => ArrayD::<i64>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Float32 => ArrayD::<f32>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Float64 => ArrayD::<f64>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Complex64 => ArrayD::<Complex<f32>>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
            DataType::Complex128 => ArrayD::<Complex<f64>>::from_shape_vec(shape, align_and_cast_buf(buf)).unwrap().into(),
        })
    }

    pub fn strides(&self) -> Vec<isize> {
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.strides().into() }
        )
    }

    pub fn to_buf(self) -> (Box<[u8]>, DataType, Box<[usize]>, Option<Box<[isize]>>) {
        let (dtype, shape) = (self.dtype(), self.shape().into_boxed_slice());

        let s = self;
        // return None if C contiguous, regular strides otherwise
        let strides = if type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.is_standard_layout() }
        ) { None } else { Some(s.strides().into_boxed_slice())};

        let buf: Box<[u8]> = type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |s| {
                align_and_cast_buf(s.into_raw_vec().into_boxed_slice()).into_boxed_slice()
            }
        );

        (buf, dtype, shape, strides)
    }

    pub fn full<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe, Sh: ShapeBuilder<Dim = IxDyn>>(shape: Sh, value: T) -> Self {
        ArrayD::<T>::from_elem(shape, value).into()
    }

    pub fn arange<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe + num::PrimInt>(start: T, end: T) -> Self {
        Array1::from_iter(num::iter::range(start, end).into_iter()).into_dyn().into()
    }

    pub fn indices<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe + num::PrimInt>(shape: &[usize], sparse: bool) -> Vec<Self> {
        shape.iter().enumerate().map(|(i, s)| {
            let arr = Array1::from_iter(num::iter::range(T::zero(), T::from(*s).unwrap()).into_iter()).into_dyn();
            let slice: Vec<SliceInfoElem> = (0..shape.len()).map(|j| if i == j { SliceInfoElem::from(..) } else { SliceInfoElem::NewAxis }).collect();
            let slice = arr.slice(&slice[..]);
            if sparse { slice } else {
                slice.broadcast(shape).unwrap()
            }.as_standard_layout().to_owned().into()
        }).collect()
    }

    pub fn linspace<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe + Float>(start: T, end: T, n: usize) -> Self {
        Array1::linspace(start, end, n).into_dyn().into()
    }

    pub fn logspace<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe + Float>(start: T, end: T, n: usize, base: T) -> Self {
        Array1::logspace(base, start, end, n).into_dyn().into()
    }

    pub fn geomspace<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe + Float>(start: T, end: T, n: usize) -> Self {
        Array1::geomspace(start, end, n).expect("Invalid bounds for geomspace").into_dyn().into()
    }

    pub fn eye<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe + Zero + One>(ndim: usize) -> Self {
        let arr: Array2<T> = Array2::eye(ndim);
        arr.into_dyn().into()
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

    pub fn broadcast_to<Sh: ShapeBuilder<Dim = IxDyn>>(self, shape: Sh) -> Result<DynArray, ShapeError> {
        // TODO because this 
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |s| {
                Ok(s.broadcast(shape.into_shape().raw_dim().clone()).ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleShape))?.to_owned().into())
            }
        )
    }

    pub fn reshape(&self, shape: &[isize]) -> Result<DynArray, String> {
        let mut used_inferred = false;
        let mut prod = 1usize;
        let size: usize = self.shape.iter().product();

        for &s in shape {
            if s == -1 {
                if used_inferred {
                    return Err(format!("Cannot use -1 on multiple dimensions in shape {:?}", shape));
                }
                used_inferred = true;
                continue;
            } else if s < 0 {
                return Err(format!("Invalid dimension '{}' in shape {:?}", s, shape));
            }
            prod = prod.checked_mul(s as usize).ok_or_else(|| format!("Overflow evaluating shape {:?}", shape))?;
        }

        let (div, rem) = size.div_rem(&prod);

        if rem != 0 || prod != size && !used_inferred {
            return Err(format!("Cannot reshape array of size {} into shape {:?}", size, shape));
        }

        let shape: Vec<usize> = shape.into_iter().map(|&v| if v < 0 { div } else { v as usize }).collect();

        let s = self;
        Ok(type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.view().into_shape(shape).unwrap().into_owned().into() }
        ))
    }

    pub fn ravel(&self) -> DynArray {
        let size = self.shape().iter().product();
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.view().into_shape(vec![size]).unwrap().as_standard_layout().to_owned().into() }
        )
    }
}

fn align_and_cast_buf<T: bytemuck::Pod, U: bytemuck::NoUninit + bytemuck::AnyBitPattern>(buf: Box<[T]>) -> Vec<U> {
    // first try to cast directly. this only works if alignment is the same
    let buf = match bytemuck::try_cast_slice_box(buf) {
        Ok(v) => return v.into_vec(),
        Err((_, buf)) => buf,
    };
    bytemuck::pod_collect_to_vec(&buf)
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

impl<T: PhysicalType + Pod + UnwindSafe + RefUnwindSafe, D: Dimension> From<Array<T, D>> for DynArray {
    fn from(value: Array<T, D>) -> Self { 
        Self::from_typed(value.into_dyn())
    }
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

#[inline]
fn mat_mul_inner<T: PhysicalType + ndarray::LinalgScalar>(lhs: &ArrayD<T>, rhs: &ArrayD<T>) -> ArrayD<T> {
    if lhs.ndim() < 1 || rhs.ndim() < 1 || lhs.shape()[lhs.ndim() - 1] != rhs.shape()[(1).min(rhs.ndim() - 1)] {
        std::panic::panic_any(ArrayError::broadcast_err(format!("Unable to matrix multiply shapes {:?} and {:?}", lhs.shape(), rhs.shape())));
    }
    let mut rhs_view = rhs.view();
    if rhs.ndim() > 1 {
        rhs_view.swap_axes(0, 1);
    }

    let lhs_size: usize = lhs.shape()[..lhs.ndim() - 1].iter().product();
    let rhs_size: usize = rhs_view.shape()[1..].iter().product();
    let shared_size: usize = rhs_view.shape()[0];
    let out_shape: Vec<usize> = lhs.shape()[..lhs.ndim() - 1].iter().chain(rhs_view.shape()[1..].iter()).copied().collect();

    let lhs_array;
    let lhs_view = if lhs.is_standard_layout() {
        lhs.view().into_shape((lhs_size, shared_size)).unwrap()
    } else {
        lhs_array = Array::from_shape_vec((lhs_size, shared_size), lhs.iter().cloned().collect()).unwrap();
        lhs_array.view()
    };

    let rhs_array;
    let rhs_view = if rhs_view.is_standard_layout() {
        rhs_view.into_shape((shared_size, rhs_size)).unwrap()
    } else {
        rhs_array = Array::from_shape_vec((shared_size, rhs_size), rhs.iter().cloned().collect()).unwrap();
        rhs_array.view()
    };

    lhs_view.dot(&rhs_view).into_shape(out_shape).unwrap()
}

#[inline]
pub(crate) fn roll_inner<T: PhysicalType>(arr: &ArrayD<T>, ax_rolls: &[isize]) -> ArrayD<T> {
    assert_eq!(arr.ndim(), ax_rolls.len());

    let shape = arr.shape();
    // for each ax, list of (input slice, output slice)
    let ax_slices: Vec<Vec<(SliceInfoElem, SliceInfoElem)>> = ax_rolls.iter().zip(shape).map(|(&roll, &size)| {
        if roll == 0 || size == 0 {
            vec![(SliceInfoElem::from(..), SliceInfoElem::from(..))]
        } else {
            let offset = roll.rem_euclid(size as isize) as usize;
            vec![
                // input slice, output slice
                (SliceInfoElem::from(..size - offset), SliceInfoElem::from(offset..)),
                (SliceInfoElem::from(size - offset..), SliceInfoElem::from(..offset)),
            ]
        }
    }).collect();

    // SAFETY: Along each axis, we either index the whole slice, or split into two sections ((offset..), (..offset)).
    // In other words, we split the array into hyperoctants. However, we assign to each of these hyperoctants.
    unsafe {
        ArrayD::build_uninit(shape, |mut out| {
            for idxs in ax_slices.into_iter().multi_cartesian_product() {
                let (in_idxs, out_idxs): (Vec<SliceInfoElem>, Vec<SliceInfoElem>) = idxs.into_iter().unzip();
                arr.slice(&in_idxs[..]).assign_to(out.slice_mut(&out_idxs[..]));
            }
        }).assume_init()
    }
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

    pub fn cast_category<'a>(&'a self, category: DataTypeCategory) -> Cow<'a, DynArray> {
        let init_dtype = self.dtype();
        if init_dtype.category() == category {
            return Cow::Borrowed(self);
        }
        let s = self;
        match init_dtype.as_category(category) {
            Some(dtype) => Cow::Owned(type_dispatch!(
                (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
                |ref s| cast_to(s, dtype).unwrap()
            )),
            None => std::panic::panic_any(ArrayError::type_err(format!("Unable to cast dtype {} to {}", init_dtype, category))),
        }
    }

    pub fn cast_min_category<'a>(&'a self, category: DataTypeCategory) -> Cow<'a, DynArray> {
        let init_dtype = self.dtype();
        let new_dtype = self.dtype().as_min_category(category);
        if init_dtype.category() == new_dtype.category() {
            return Cow::Borrowed(self);
        }
        self.cast(new_dtype)
    }

    pub fn roll(&self, rolls: &[isize], axes: &[isize]) -> DynArray {
        if rolls.len() != axes.len() {
            std::panic::panic_any(ArrayError::value_err("'rolls' must be same length as 'axes'"));
        }
        let axes: Vec<usize> = axes.iter().map(|ax| normalize_axis(*ax, self.ndim())).collect();

        let mut ax_rolls: Vec<isize> = vec![0; self.ndim()];
        for (roll, ax) in rolls.iter().zip(axes) {
            ax_rolls[ax] = *roll;
        }

        let s = self;
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| roll_inner(s, &ax_rolls).into()
        )
    }

    pub fn abs(&self) -> DynArray {
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

    pub fn exp(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.exp()).into() },
        )
    }

    pub fn sqrt(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.sqrt()).into() },
        )
    }

    pub fn ceil(&self) -> DynArray {
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { s.clone().into() },
            (f32, f64),
            |ref s| { s.mapv(|e| e.ceil()).into() },
        )
    }

    pub fn floor(&self) -> DynArray {
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { s.clone().into() },
            (f32, f64),
            |ref s| { s.mapv(|e| e.floor()).into() },
        )
    }

    pub fn conj(&self) -> DynArray {
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64),
            |ref s| { s.clone().into() },
            (Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.conj()).into() },
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

    pub fn pow<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        // TODO more granular dispatch here
        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| l.wrapping_pow(*r as u32)).into() },
            (f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| l.powf(*r)).into() },
            (Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| l.powc(*r)).into() },
        )
    }

    pub fn mat_mul<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();
        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = (self.cast(ty), rhs.cast(ty));
        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { mat_mul_inner(lhs, rhs).into() }
        )
    }

    pub fn apply_cmap(&self) -> DynArray {
        let s = self;
        type_dispatch!(
            (f32, f64),
            |ref s| { apply_cmap_u8(magma(), s.view()).into() }
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
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
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