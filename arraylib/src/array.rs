use std::any::Any;
use std::ops;
use std::borrow::{Borrow, Cow};
use std::fmt;
use std::panic::{RefUnwindSafe, UnwindSafe};

use bytemuck::Pod;
use itertools::{Itertools, izip};
use num::{Float, Zero, One, Integer};
use num_complex::{Complex, ComplexFloat};
use ordered_float::NotNan;
use ndarray::{Array, Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, Axis};
use ndarray::{Dimension, ErrorKind, IxDyn, ShapeBuilder, ShapeError, SliceInfoElem, Zip};

use arraylib_macro::{type_dispatch, forward_val_to_ref};
use crate::dtype::{DataType, DataTypeCategory, PhysicalType, Bool, IsClose, promote_types};
use crate::cast::Cast;
use crate::error::ArrayError;
use crate::colors::apply_cmap_u8;
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

    pub fn size(&self) -> usize { self.shape.iter().product() }

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

    pub fn meshgrid(arrs: Vec<DynArray>, sparse: bool) -> Result<Vec<DynArray>, String> {
        let mut shape: Vec<usize> = Vec::new();

        for arr in &arrs {
            if arr.ndim() != 1 {
                return Err("'meshgrid' requires 1D input arrays".to_owned());
            }
            shape.push(arr.size())
        }

        Ok(arrs.into_iter().enumerate().map(|(i, arr)| {
            type_dispatch!(
                (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
                |ref arr| {
                    let slice: Vec<SliceInfoElem> = (0..shape.len()).map(|j| if i == j { SliceInfoElem::from(..) } else { SliceInfoElem::NewAxis }).collect();
                    let slice = arr.slice(&slice[..]);
                    if sparse { slice } else {
                        slice.broadcast(shape.clone()).unwrap()
                    }.as_standard_layout().to_owned().into()
                }
            )
        }).collect())
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

    pub fn broadcast_to<Sh: ShapeBuilder<Dim = IxDyn>>(&self, shape: Sh) -> Result<DynArray, ShapeError> {
        // TODO because this 
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| {
                Ok(s.broadcast(shape.into_shape().raw_dim().clone()).ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleShape))?.to_owned().into())
            }
        )
    }

    pub fn reshape(&self, shape: &[isize]) -> Result<DynArray, String> {
        let mut used_inferred = false;
        let mut prod = 1usize;
        let size: usize = self.size();

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
        let size = self.size();
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.view().into_shape(vec![size]).unwrap().as_standard_layout().to_owned().into() }
        )
    }

    pub fn try_mul<A: Borrow<DynArray>>(&self, rhs: A) -> Result<DynArray, String> {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).map_err(|e| e.to_string())?;
        Ok(type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1.wrapping_mul(e2)).into() },
            (Bool, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 * e2).into() },
        ))
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

fn broadcast_err(shapes: &[&[usize]]) -> String {
    use fmt::Write;

    let mut buf: String = "Unable to broadcast shapes ".to_owned();
    assert!(shapes.len() >= 2);
    for (i, shape) in shapes.iter().enumerate() {
        if i == shapes.len() - 1 {
            write!(&mut buf, "and {:?}", shape).unwrap();
        } else if shapes.len() > 2 {
            write!(&mut buf, "{:?}, ", shape).unwrap();
        } else {
            write!(&mut buf, "{:?} ", shape).unwrap();
        }
    }
    buf
}

pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, String> {
    let mut out_shape: Vec<usize> = Vec::new();

    let mut iters: Vec<_> = shapes.iter().map(|sh| sh.iter().rev()).collect();

    'dim: loop {
        let mut shapes_iter = iters.iter_mut();

        let mut len = loop {
            match shapes_iter.next() {
                Some(it) => match it.next() {
                    None => continue,
                    Some(&val) => break val,
                },
                None => break 'dim,
            }
        };

        for it in shapes_iter {
            match it.next() {
                None | Some(1) => (),
                Some(&v) => if v != len {
                    if len == 1 {
                        len = v;
                    } else {
                        return Err(broadcast_err(shapes));
                    }
                }
            }
        }

        out_shape.push(len);
    }

    out_shape.reverse();
    Ok(out_shape)
}

pub fn broadcast_arrays(arrs: &[&DynArray]) -> Result<Vec<DynArray>, String> {
    let shapes: Vec<&[usize]> = arrs.iter().map(|arr| arr.shape.as_slice()).collect();

    let shape = broadcast_shapes(&shapes)?;

    Ok(arrs.iter().map(|arr| arr.broadcast_to(shape.as_slice()).unwrap()).collect())
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
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).expect("Failed to broadcast");
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
impl<'a> ops::Not for &'a DynArray {
    type Output = DynArray;

    fn not(self) -> Self::Output {
        let s = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref s| { s.mapv(|v| !v).into() }
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
impl<'a, T: Borrow<DynArray>> ops::BitOr<T> for &'a DynArray {
    type Output = DynArray;

    fn bitor(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 | e2).into() },
        )
    }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::BitXor<T> for &'a DynArray {
    type Output = DynArray;

    fn bitxor(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 ^ e2).into() },
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

pub fn stack<'a, I: IntoIterator<Item = &'a DynArray>>(arrs: I, axis: isize) -> Result<DynArray, String> {
    let arrs = arrs.into_iter().collect_vec();

    let dtypes: Vec<_> = arrs.iter().map(|arr| arr.dtype()).collect();
    let dtype = promote_types(dtypes.as_slice());

    let ax = normalize_axis(axis, arrs[0].ndim() + 1);

    let arrs: Vec<_> = arrs.into_iter().map(|arr| arr.cast(dtype)).collect();

    match dtype {
        DataType::Boolean => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<Bool>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Int8 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<i8>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Int16 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<i16>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Int32 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<i32>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Int64 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<i64>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::UInt8 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<u8>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::UInt16 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<u16>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::UInt32 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<u32>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::UInt64 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<u64>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Float32 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<f32>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Float64 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<f64>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Complex64 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<Complex<f32>>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
        DataType::Complex128 => {
            let v: Vec<_> = arrs.iter().map(|arr| arr.downcast_ref::<Complex<f64>>().unwrap().view()).collect();
            ndarray::stack(Axis(ax), v.as_slice()).map(|arr| arr.into())
        },
    }.map_err(|e| format!("{}", e))
}

fn as_not_nan<'a, T: ordered_float::FloatCore>(slice: &'a [T]) -> Option<&'a [NotNan<T>]> {
    if slice.iter().all(|v| !v.is_nan()) {
        unsafe { Some(std::mem::transmute(slice)) }
    } else {
        None
    }
}

pub fn interp(xs: &'_ DynArray, xp: &'_ DynArray, yp: &'_ DynArray, left: Option<f64>, right: Option<f64>) -> Result<DynArray, String> {
    let dtype = promote_types(&[xs.dtype(), xp.dtype(), yp.dtype().real_dtype()]).as_min_category(DataTypeCategory::Floating);
    let y_complex = yp.dtype().category() == DataTypeCategory::Complex;

    if xp.dtype().category() == DataTypeCategory::Complex {
        return Err("'xs' and 'xp' must not be complex".to_owned());
    }

    if xp.ndim() != 1 || yp.ndim() != 1 || xp.size() != yp.size() {
        return Err("'xp' and 'yp' must be 1D arrays of the same length".to_owned());
    }
    if xp.size() < 2 || yp.size() < 2 {
        return Err("Expected at least two coordinates to interpolate".to_owned());
    }

    let (xs, xp, yp) = (xs.cast(dtype), xp.cast(dtype), if y_complex { Cow::Borrowed(yp) } else { yp.cast(dtype) });

    type_dispatch!(
        (f32,),
        |ref xs, ref xp| {
            if y_complex {
                interp_inner(
                    xs.view(), xp.view().into_dimensionality().unwrap(), yp.downcast_ref::<Complex<f32>>().unwrap().view().into_dimensionality().unwrap(),
                    left.map(|v| Complex::<f32>::from(v as f32)), right.map(|v| Complex::<f32>::from(v as f32))
                ).map(|arr| arr.into())
            } else {
                interp_inner(
                    xs.view(), xp.view().into_dimensionality().unwrap(), yp.downcast_ref::<f32>().unwrap().view().into_dimensionality().unwrap(),
                    left.map(|v| v as f32), right.map(|v| v as f32)
                ).map(|arr| arr.into())
            }
        },
        (f64,),
        |ref xs, ref xp| {
            if y_complex {
                interp_inner(
                    xs.view(), xp.view().into_dimensionality().unwrap(), yp.downcast_ref::<Complex<f64>>().unwrap().view().into_dimensionality().unwrap(),
                    left.map(Complex::<f64>::from), right.map(Complex::<f64>::from)
                ).map(|arr| arr.into())
            } else {
                interp_inner(
                    xs.view(), xp.view().into_dimensionality().unwrap(), yp.downcast_ref::<f64>().unwrap().view().into_dimensionality().unwrap(),
                    left.map(|v| v as f64), right.map(|v| v as f64)
                ).map(|arr| arr.into())
            }
        },
    )
}

fn interp_inner<T: PhysicalType + ordered_float::FloatCore + Into<U>, U: PhysicalType + ComplexFloat>(xs: ArrayViewD<'_, T>, xp: ArrayView1<'_, T>, yp: ArrayView1<'_, U>, left: Option<U>, right: Option<U>) -> Result<ArrayD<U>, String> {
    // we've already checked that xp and yp are the same length and len >2
    // get slices of fp and xp in standard order
    let xp_store: Vec<T>;
    let xp = if let Some(s) = xp.as_slice() {
        s
    } else {
        xp_store = xp.iter().copied().collect();
        xp_store.as_slice()
    };
    let xp = as_not_nan(xp).ok_or_else(|| "'xp' must not contain NaN values".to_owned())?;

    let yp_store: Vec<U>;
    let yp = if let Some(s) = yp.as_slice() {
        s
    } else {
        yp_store = yp.iter().copied().collect();
        yp_store.as_slice()
    };

    // SAFETY: we've checked the size of xp and fp
    let (x_left, x_right) = unsafe { (xp.get_unchecked(0).into_inner(), xp.get_unchecked(xp.len() - 1).into_inner()) };
    let (y_left, y_right) = unsafe { (left.unwrap_or(*yp.get_unchecked(0)), right.unwrap_or(*yp.get_unchecked(yp.len() - 1))) };

    let n_xs: usize = xs.shape().iter().product();

    let slopes: Option<Array1<U>> = if n_xs >= xp.len() { Some(
        (0..xp.len() - 1).map(|i|
            // SAFETY: xp and fp are the same length, and we only index at max len-1
            unsafe { (*yp.get_unchecked(i+1) - *yp.get_unchecked(i)) / (xp.get_unchecked(i+1).into_inner() - xp.get_unchecked(i).into_inner()).into() }
        ).collect()
    )} else { None };

    Ok(xs.mapv(|x| {
        if x <= x_left { y_left }
        else if x >= x_right { y_right }
        else if x.is_nan() { x.into() }
        else {
            //binary search
            // SAFETY: we checked x is not NaN
            match xp.binary_search(unsafe { &NotNan::new_unchecked(x) }) {
                Ok(j) => {
                    // exact match
                    yp[j]
                },
                Err(j) => {
                    // interpolate between j-1 and j
                    let slope = if let Some(slopes) = &slopes {
                        slopes[j-1]
                    } else {
                        (yp[j] - yp[j-1]) / (xp[j].into_inner() - xp[j-1].into_inner()).into()
                    };
                    (x - xp[j-1].into_inner()).into() * slope + yp[j-1]
                }
            }
        }
    }))
}

pub fn interpn(coords: &[&'_ DynArray], values: &'_ DynArray, xs: &'_ DynArray, fill: Option<f64>) -> Result<DynArray, String> {
    let mut dtypes: Vec<DataType> = coords.iter().map(|arr| arr.dtype()).collect();
    dtypes.push(values.dtype().real_dtype());
    dtypes.push(xs.dtype());
    let dtype = promote_types(dtypes.as_slice()).as_min_category(DataTypeCategory::Floating);
    if dtype.category() == DataTypeCategory::Complex {
        return Err("'coords' and 'xs' must not be complex".to_owned());
    }
    let values_complex = values.dtype().category() == DataTypeCategory::Complex;

    let expected_values_shape: Vec<usize> = coords.iter().map(|coords| {
        if coords.ndim() != 1 {
            Err("'coords' must be a list of 1D arrays".to_owned())
        } else {
            Ok(coords.size())
        }
    }).try_collect()?;

    if values.shape() != expected_values_shape {
        return Err(format!("'values' must match the shape of 'coords': {:?}", expected_values_shape.as_slice()));
    }

    if xs.ndim() < 1 || xs.shape()[xs.ndim() - 1] != coords.len() {
        return Err(format!("'xs' must be an array of shape [..., {}]", coords.len()));
    }

    //let (xs, xp, yp) = (xs.cast(dtype), xp.cast(dtype), if y_complex { Cow::Borrowed(yp) } else { yp.cast(dtype) });
    let coords: Vec<_> = coords.into_iter().map(|arr| arr.cast(dtype)).collect();
    let (values, xs) = (values.cast(dtype), xs.cast(dtype));

    match dtype {
        DataType::Float32 => {
            if values_complex {
                interpn_inner(
                    coords.iter().map(|coords| coords.downcast_ref::<f32>().unwrap().view().into_dimensionality().unwrap()).collect(),
                    values.downcast_ref::<Complex<f32>>().unwrap().view(),
                    xs.downcast_ref::<f32>().unwrap().view(),
                    fill.map(|val| (val as f32).into()),
                ).map(|arr| arr.into())
            } else {
                interpn_inner(
                    coords.iter().map(|coords| coords.downcast_ref::<f32>().unwrap().view().into_dimensionality().unwrap()).collect(),
                    values.downcast_ref::<f32>().unwrap().view(),
                    xs.downcast_ref::<f32>().unwrap().view(),
                    fill.map(|val| val as f32),
                ).map(|arr| arr.into())
            }
        },
        DataType::Float64 => {
            if values_complex {
                interpn_inner(
                    coords.iter().map(|coords| coords.downcast_ref::<f64>().unwrap().view().into_dimensionality().unwrap()).collect(),
                    values.downcast_ref::<Complex<f64>>().unwrap().view(),
                    xs.downcast_ref::<f64>().unwrap().view(),
                    fill.map(|val| (val as f64).into()),
                ).map(|arr| arr.into())
            } else {
                interpn_inner(
                    coords.iter().map(|coords| coords.downcast_ref::<f64>().unwrap().view().into_dimensionality().unwrap()).collect(),
                    values.downcast_ref::<f64>().unwrap().view(),
                    xs.downcast_ref::<f64>().unwrap().view(),
                    fill.map(|val| val as f64),
                ).map(|arr| arr.into())
            }
        },
        _ => unreachable!()
    }
}

fn interpn_inner<T, U>(coords: Vec<ArrayView1<'_, T>>, values: ArrayViewD<'_, U>, xs: ArrayViewD<'_, T>, fill: Option<U>) -> Result<ArrayD<U>, String>
where T: PhysicalType + Float + Into<U> + std::iter::Product + std::fmt::Debug,
      U: PhysicalType + ComplexFloat + std::fmt::Debug
{
    let coords: Vec<Cow<'_, [T]>> = coords.iter().map(|a| {
        if let Some(s) = a.as_slice() { Cow::Borrowed(s) } else {
            Cow::Owned(a.iter().copied().collect::<Vec<_>>())
        }
    }).collect();

    if xs.ndim() == 0 || xs.shape()[xs.ndim() - 1] != coords.len() {
        return Err(format!("'xs' must be an array of shape [..., {}]", coords.len()));
    }

    let mut filled = false;
    let dim_ax: Axis = Axis(xs.ndim() - 1);

    let fill_val = fill.unwrap_or(U::zero());

    let result = xs.map_axis(dim_ax, |x_vals| {
        let mut idxs: Vec<usize> = Vec::with_capacity(x_vals.len());
        for (coords, &x_val) in coords.iter().zip(x_vals) {
            //log::log(format!("looking for {:?} in coords {:?}", x_val, coords));
            if x_val < *coords.first().unwrap() || x_val >= *coords.last().unwrap() {
                if x_val == *coords.last().unwrap() {
                    idxs.push(coords.len() - 2);
                } else {
                    filled = true;
                    return fill_val;
                }
            } else if x_val.is_nan() {
                return U::from(T::nan()).unwrap();
            } else {
                idxs.push(coords.partition_point(|&v| v < x_val) - 1);
            }
        }

        //log::log(format!("idxs: {:?}", &idxs));

        let coord_diffs: Vec<(T, T)> = izip!(&coords, &idxs, x_vals)
            .map(|(coords, &i, &x)| (coords[i+1] - x, x - coords[i])).collect();

        //log::log(format!("coord_diffs: {:?}", &coord_diffs));

        let mut out: U = U::zero();
        let mut weights: T = T::zero();

        for idx_diffs in idxs.iter().map(|_| [0, 1].into_iter()).multi_cartesian_product() {
            let val: U = values[idxs.iter().zip(&idx_diffs).map(|(i, d)| i + d).collect_vec().as_slice()];

            let weight: T = idx_diffs.iter().zip(&coord_diffs).map(|(&i, d)| if i == 0 { d.0 } else { d.1 }).product();
            //log::log(format!("val: {:?} weight: {:?}", val, weight));
            weights = weights + weight;
            out = out + val * weight.into();
        }
        out / weights.into()
    });

    if let None = fill {
        if filled {
            return Err("Out-of-bounds points and no fill value specified".to_owned());
        }
    }

    Ok(result)
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
            |ref s| { s.map(|e| e.exp()).into() },
        )
    }

    pub fn sqrt(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.map(|e| e.sqrt()).into() },
        )
    }

    // trig functions

    pub fn sin(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.map(|e| e.sin()).into() },
        )
    }

    pub fn cos(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.map(|e| e.cos()).into() },
        )
    }

    pub fn tan(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.map(|e| e.tan()).into() },
        )
    }

    pub fn arcsin(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.asin()).into() },
        )
    }

    pub fn arccos(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.acos()).into() },
        )
    }

    pub fn arctan(&self) -> DynArray {
        let s = self.cast_min_category(DataTypeCategory::Floating);
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| { s.mapv(|e| e.atan()).into() },
        )
    }

    pub fn arctan2<A: Borrow<DynArray>>(&self, other: A) -> DynArray {
        let other = other.borrow();
        let ty = promote_types(&[self.dtype, other.dtype]).as_min_category(DataTypeCategory::Floating);
        let (y, x) = self.cast(ty).broadcast_with(&other.cast(ty)).unwrap();
        type_dispatch!(
            (f32, f64),
            |ref y, ref x| { Zip::from(y).and(x).map_collect(|y, x| y.atan2(*x) ).into() },
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
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l == r)).into() }
        )
    }

    pub fn not_equals<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l != r)).into() }
        )
    }

    pub fn isclose<T: Borrow<DynArray>>(&self, other: T, rtol: f64, atol: f64) -> DynArray {
        use num::FromPrimitive;

        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l == r)).into() },
            (f32, Complex<f32>),
            |ref lhs, ref rhs| {
                let rtol = f32::from_f64(rtol).unwrap();
                let atol = f32::from_f64(atol).unwrap();

                Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l.is_close(*r, rtol, atol))).into()
            },
            (f64, Complex<f64>),
            |ref lhs, ref rhs| {
                Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l.is_close(*r, rtol, atol))).into()
            }
        )
    }

    pub fn is_normal(&self) -> DynArray {
        let s = self;
        type_dispatch!(
            (f32, f64, Complex<f32>, Complex<f64>),
            |ref s| s.map(|v| Bool::from(v.is_normal())).into()
        )
    }

    pub fn all<T: Borrow<DynArray>>(&self) -> bool {
        let arr = self.cast(DataType::Boolean);
        let arr_ref = arr.downcast_ref::<Bool>().unwrap();
        if let Some(slice) = arr_ref.as_slice_memory_order() {
            slice.iter().all(|v| (*v).into())
        } else {
            arr_ref.iter().all(|v| (*v).into())
        }
    }

    pub fn any<T: Borrow<DynArray>>(&self) -> bool {
        let arr = self.cast(DataType::Boolean);
        let arr_ref = arr.downcast_ref::<Bool>().unwrap();
        if let Some(slice) = arr_ref.as_slice_memory_order() {
            slice.iter().any(|v| (*v).into())
        } else {
            arr_ref.iter().any(|v| (*v).into())
        }
    }

    pub fn allequal<T: Borrow<DynArray>>(&self, other: T) -> bool {
        let arr = self.equals(other).downcast::<Bool>().expect("'equals' returned wrong type");
        arr.into_raw_vec().into_iter().all(|b| b.into())
    }

    pub fn allclose<T: Borrow<DynArray>>(&self, other: T, rtol: f64, atol: f64) -> bool {
        let arr = self.isclose(other, rtol, atol).downcast::<Bool>().unwrap();
        arr.into_raw_vec().into_iter().all(|b| b.into())
    }

    pub fn less<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l < r)).into() }
        )
    }

    pub fn less_equal<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l <= r)).into() }
        )
    }

    pub fn greater<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l > r)).into() }
        )
    }

    pub fn greater_equal<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l >= r)).into() }
        )
    }

    pub fn minimum<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l <= r { l } else { r }
            ).into() },
            (f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    if l.is_nan() { l } else { r }
                } else if l <= r { l } else { r }
            ).into() },
            (Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    if l.is_nan() { l } else { r }
                } else if l.abs() <= r.abs() { l } else { r }
            ).into() },
        )
    }

    pub fn maximum<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l >= r { l } else { r }
            ).into() },
            (f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    // return whichever is NaN, or first
                    if l.is_nan() { l } else { r }
                } else if l >= r { l } else { r }
            ).into() },
            (Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    if l.is_nan() { l } else { r }
                } else if l.abs() >= r.abs() { l } else { r }
            ).into() },
        )
    }

    pub fn nanminimum<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l <= r { l } else { r }
            ).into() },
            (f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    // return whichever is not NaN, or first
                    if r.is_nan() { l } else { r }
                } else if l <= r { l } else { r }
            ).into() },
            (Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    if r.is_nan() { l } else { r }
                } else if l.abs() <= r.abs() { l } else { r }
            ).into() },
        )
    }

    pub fn nanmaximum<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let rhs = other.borrow();

        let ty = promote_types(&[self.dtype, rhs.dtype]);
        let (lhs, rhs) = self.cast(ty).broadcast_with(&rhs.cast(ty)).unwrap();

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l >= r { l } else { r }
            ).into() },
            (f32, f64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    // return whichever is not NaN, or first
                    if r.is_nan() { l } else { r }
                } else if l >= r { l } else { r }
            ).into() },
            (Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&l, &r|
                if l.is_nan() || r.is_nan() {
                    if r.is_nan() { l } else { r }
                } else if l.abs() >= r.abs() { l } else { r }
            ).into() },
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

    pub fn apply_cmap(&self,
        cmap: ArrayView2<'static, f32>,
        min_color: Option<ArrayView1<'_, f32>>,
        max_color: Option<ArrayView1<'_, f32>>,
        invalid_color: ArrayView1<'_, f32>,
    ) -> DynArray {
        let s = self;
        type_dispatch!(
            (f32, f64),
            |ref s| { apply_cmap_u8(cmap, s.view(), min_color, max_color, invalid_color).into() }
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