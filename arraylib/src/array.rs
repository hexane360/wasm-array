use std::any::Any;
use std::ops;
use std::borrow::Borrow;

use bytemuck::Pod;
use num_complex::{Complex, ComplexFloat};
use ndarray::{Array, ArrayD, Zip, Dimension};

use arraylib_macro::{type_dispatch, forward_val_to_ref};
use crate::dtype::{DataType, PhysicalType, Bool};

#[derive(Debug)]
pub struct DynArray {
    dtype: DataType,
    inner: Box<dyn Any>,
}

impl DynArray {
    pub fn dtype(&self) -> DataType { self.dtype }

    pub fn from_typed<T: PhysicalType + Pod + 'static>(arr: ArrayD<T>) -> Self {
        Self {
            dtype: T::DATATYPE,
            inner: Box::new(arr) as Box<dyn Any>,
        }
    }

    pub fn downcast<T: PhysicalType + Pod + 'static>(self) -> Option<ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(*self.inner.downcast().unwrap())
    }

    pub fn downcast_ref<T: PhysicalType + Pod + 'static>(&self) -> Option<&ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(self.inner.downcast_ref().unwrap())
    }

    pub fn downcast_mut<T: PhysicalType + Pod + 'static>(&mut self) -> Option<&mut ArrayD<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(self.inner.downcast_mut().unwrap())
    }

}

impl<T: PhysicalType + Pod + 'static, D: Dimension> From<Array<T, D>> for DynArray {
    fn from(value: Array<T, D>) -> Self { Self::from_typed(value.into_dyn()) }
}

#[forward_val_to_ref]
impl<'a, T: Borrow<DynArray>> ops::Add<T> for &'a DynArray {
    type Output = DynArray;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.borrow();

        assert!(self.dtype == rhs.dtype);

        let lhs = self;
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

        assert!(self.dtype == rhs.dtype);

        let lhs = self;
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

        assert!(self.dtype == rhs.dtype);

        let lhs = self;
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

        assert!(self.dtype == rhs.dtype);

        let lhs = self;
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

        assert!(self.dtype == rhs.dtype);

        let lhs = self;
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|&e1, &e2| e1 & e2).into() },
        )
    }
}

impl<'a> ops::AddAssign<&'a DynArray> for DynArray {
    fn add_assign(&mut self, other: &'a DynArray) {
        assert!(self.dtype == other.dtype);

        let (lhs, rhs) = (self, other);
        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref mut lhs, ref rhs| { lhs.zip_mut_with(rhs, |e1, e2| *e1 += *e2) }
        );
    }
}

impl ops::AddAssign<DynArray> for DynArray {
    fn add_assign(&mut self, rhs: DynArray) { self.add_assign(&rhs) }
}

impl<T: Borrow<DynArray>> ops::SubAssign<T> for DynArray {
    fn sub_assign(&mut self, other: T) {
        let (lhs, rhs) = (self, other.borrow());

        assert!(lhs.dtype == rhs.dtype);

        type_dispatch!(
            (Bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref mut lhs, ref rhs| { lhs.zip_mut_with(rhs, |e1, e2| *e1 -= *e2) }
        );
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

impl DynArray {
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
        let (lhs, rhs) = (self, other.borrow());

        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l == r)).into() }
        )
    }

    pub fn not_equals<T: Borrow<DynArray>>(&self, other: T) -> DynArray {
        let (lhs, rhs) = (self, other.borrow());

        type_dispatch!(
            (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
            |ref lhs, ref rhs| { Zip::from(lhs).and(rhs).map_collect(|l, r| Bool::from(l != r)).into() }
        )
    }
}