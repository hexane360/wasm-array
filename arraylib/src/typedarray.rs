use std::{mem::{self, MaybeUninit}, num::NonZeroU8};
use std::ops;

use zerocopy::{FromBytes, AsBytes};

use crate::dtype::{DataType, PhysicalType};
use crate::array::Array;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides {
    strides: Box<[usize]>,
    shape: Box<[usize]>,
}

impl Strides {
    pub fn size(&self) -> usize {
        let mut size: usize = 0;
            for (&stride, &shape) in self.strides.iter().zip(self.shape.iter()) {
                size += stride * shape;
            }
        size
    }

    fn checked_size(&self) -> Option<usize> {
        let mut size: usize = 0;
            for (&stride, &shape) in self.strides.iter().zip(self.shape.iter()) {
                size = size.checked_add(stride.checked_mul(shape)?)?;
            }
        Some(size)
    }
}

/// INVARIANTS:
/// strides.size() must equal buf.len()
#[derive(Clone)]
pub struct TypedArray<T: PhysicalType> {
    strides: Strides,
    buf: Box<[T]>,
}

impl<T> TypedArray<T> where T: PhysicalType {
    pub fn empty(strides: Strides) -> TypedArray<MaybeUninit<T>> {
        let size = strides.checked_size().expect("strides overflow usize");
        let buf = Box::new_uninit_slice(size);
        TypedArray {
            strides,
            buf,
        }
    }

    pub fn zeroed(strides: Strides) -> Self {
        let size = strides.checked_size().expect("strides overflow usize");
        let buf = T::new_box_slice_zeroed(size);
        Self {
            strides,
            buf,
        }
    }
}

impl<T: PhysicalType + AsBytes + 'static> TypedArray<T> {
    pub fn type_erase(self) -> Array { Array::from_typed(self) }
}

impl<T: PhysicalType> TypedArray<MaybeUninit<T>> {
    pub unsafe fn assume_init(self) -> TypedArray<T> {
        TypedArray {
            strides: self.strides,
            buf: self.buf.assume_init(),
        }
    }
}

pub trait ArrRef<'a, T> where T: PhysicalType {
    fn strides(&self) -> &Strides;
    fn buf_ref(&self) -> &[T];

    fn size(&self) -> usize { self.strides().size() }
    fn shape(&self) -> &[usize] { &self.strides().shape }

    fn elementwise_unary_ref<U, F>(&self, f: F) -> TypedArray<U>
        where F: Fn(&T) -> U,
              U: PhysicalType
    {
        let mut out = TypedArray::empty(self.strides().clone());

        for (v, o) in self.buf_ref().iter().zip(out.buf_ref_mut()) {
            unsafe { std::ptr::write(o, MaybeUninit::new(f(v))); }
        }

        unsafe { out.assume_init() }
    }

    fn elementwise_binary_ref<U, V, F>(&self, other: &TypedArray<U>, f: F) -> TypedArray<V> 
        where F: Fn(&T, &U) -> V,
              U: PhysicalType,
              V: PhysicalType
    {
        assert_eq!(self.strides(), other.strides());

        let mut out = TypedArray::empty(self.strides().clone());

        for ((l, r), o) in self.buf_ref().iter().zip(other.buf_ref()).zip(out.buf_ref_mut()) {
            unsafe { std::ptr::write(o, MaybeUninit::new(f(l, r))); }
        }

        unsafe { out.assume_init() }
    }
}

pub trait ArrRefMut<'a, T>: ArrRef<'a, T> where T: PhysicalType {
    fn buf_ref_mut(&mut self) -> &mut [T];

    fn elementwise_unary_inplace<U, F>(&mut self, f: F)
        where F: Fn(&mut T) -> ()
    {
        self.buf_ref_mut().iter_mut().for_each(f)
    }

    fn elementwise_binary_inplace<U, F>(&mut self, other: &TypedArray<U>, f: F)
        where F: Fn(&mut T, &U) -> (), U: PhysicalType
    {
        assert_eq!(self.strides(), other.strides());

        self.buf_ref_mut().iter_mut().zip(other.buf_ref())
            .for_each(|(l, r)| f(l, r));
    }
}

pub trait ArrOwned<T>: ArrRefMut<'static, T> + Sized where T: PhysicalType {
    fn into_raw(self) -> (Strides, Box<[T]>);

    fn elementwise_unary<U, F>(self, f: F) -> TypedArray<U> 
        where F: Fn(T) -> U,
              U: PhysicalType,
    {
        let mut out = TypedArray::empty(self.strides().clone());

        for (i, o) in Vec::from(self.into_raw().1).into_iter().zip(out.buf_ref_mut()) {
            unsafe { std::ptr::write(o, MaybeUninit::new(f(i))); }
        }

        unsafe { out.assume_init() }
    }

    fn elementwise_binary<U, V, F>(self, other: &TypedArray<U>, f: F) -> TypedArray<V> 
        where F: Fn(T, &U) -> V,
              U: PhysicalType,
              V: PhysicalType
    {
        assert_eq!(self.strides(), other.strides());

        let mut out = TypedArray::empty(self.strides().clone());

        for ((l, r), o) in Vec::from(self.into_raw().1).into_iter().zip(other.buf_ref()).zip(out.buf_ref_mut()) {
            unsafe { std::ptr::write(o, MaybeUninit::new(f(l, r))); }
        }

        unsafe { out.assume_init() }
    }
}

impl<T> ArrRef<'static, T> for TypedArray<T> where T: PhysicalType {
    fn strides(&self) -> &Strides { &self.strides }
    fn buf_ref(&self) -> &[T] { &self.buf }
}

impl<T> ArrRefMut<'static, T> for TypedArray<T> where T: PhysicalType {
    fn buf_ref_mut(&mut self) -> &mut [T] { &mut self.buf }
}

impl<T> ArrOwned<T> for TypedArray<T> where T: PhysicalType {
    fn into_raw(self) -> (Strides, Box<[T]>) { (self.strides, self.buf.into()) }
}

impl<'a, T> ArrRef<'a, T> for &'a TypedArray<T> where T: PhysicalType {
    fn strides(&self) -> &Strides { &self.strides }
    fn buf_ref(&self) -> &[T] { &self.buf }
}

impl<'a, T> ArrRef<'a, T> for &'a mut TypedArray<T> where T: PhysicalType {
    fn strides(&self) -> &Strides { &self.strides }
    fn buf_ref(&self) -> &[T] { &self.buf }
}

impl<'a, T> ArrRefMut<'a, T> for &'a mut TypedArray<T> where T: PhysicalType {
    fn buf_ref_mut(&mut self) -> &mut [T] { &mut self.buf }
}

fn slice_box_to_bytes<T: PhysicalType + AsBytes>(input: Box<[T]>) -> Box<[u8]> {
    let len = mem::size_of::<T>() * input.len(); // won't overflow, as `input` exists
    let ptr = Box::into_raw(input) as *mut u8;

    // SAFETY: [T] is AsBytes, ensuring that it is initialized and
    // can be interpreted as a [u8] of size size_of::<T>() * N.
    // We don't need to check for alignment since [u8] has alignment 1.
    unsafe {
        Box::<[u8]>::from_raw(core::slice::from_raw_parts_mut(ptr, len))
    }
}

fn bytes_to_slice_box<T: PhysicalType + FromBytes>(input: Box<[u8]>) -> Box<[T]> {
    if input.len() % mem::size_of::<T>() != 0 || input.as_ptr() as usize % mem::align_of::<T>() != 0 {
        panic!("Unaligned or improperly sized memory")
    }

    let len = input.len() / mem::size_of::<T>();
    let ptr = Box::into_raw(input) as *mut T;

    // SAFETY: [T] is FromBytes, ensuring that it can be initialized from any bit pattern
    // Further, we've ensured that `input` is properly aligned and of the right size
    unsafe {
        Box::<[T]>::from_raw(core::slice::from_raw_parts_mut(ptr, len))
    }
}

macro_rules! impl_binary_op {
    ($first:ident$(::$rest:ident)*, $fn:ident) => {
        // implement on reference
        impl<'a, 'b, T> $first$(::$rest)*::<&'b TypedArray<T>> for &'a TypedArray<T> 
        where T: PhysicalType + FromBytes + Copy + $first$(::$rest)*::<Output = T>,
        {
            type Output = TypedArray<T>;

            fn $fn(self, rhs: &'b TypedArray<T>) -> TypedArray<T> {
                self.elementwise_binary_ref(rhs, |l, r| <T as $first$(::$rest)*>::$fn(*l, *r))
            }
        }

        // and on owned type
        impl<T> $first$(::$rest)*::<&TypedArray<T>> for TypedArray<T> 
        where T: PhysicalType + FromBytes + Copy + $first$(::$rest)*::<Output = T>,
        {
            type Output = Self;

            fn $fn(self, rhs: &Self) -> Self {
                self.elementwise_binary(rhs, |l, r| <T as $first$(::$rest)*>::$fn(l, *r))
            }
        }
    };
}

impl_binary_op!(ops::Add, add);
impl_binary_op!(ops::Sub, sub);
impl_binary_op!(ops::Mul, mul);
impl_binary_op!(ops::Div, div);
impl_binary_op!(ops::Rem, rem);
impl_binary_op!(ops::BitAnd, bitand);
impl_binary_op!(ops::BitOr, bitor);
impl_binary_op!(ops::BitXor, bitxor);
impl_binary_op!(ops::Shl, shl);
impl_binary_op!(ops::Shr, shr);

macro_rules! impl_binary_op_inplace {
    ($first:ident$(::$rest:ident)*, $fn:ident) => {
        impl<T> $first$(::$rest)*::<&TypedArray<T>> for TypedArray<T> 
        where T: PhysicalType + FromBytes + Copy + $first$(::$rest)*,
        {
            fn $fn(&mut self, rhs: &Self) {
                self.elementwise_binary_inplace(rhs, |l, r| <T as $first$(::$rest)*>::$fn(l, *r))
            }
        }
    };
}

impl_binary_op_inplace!(ops::AddAssign, add_assign);
impl_binary_op_inplace!(ops::SubAssign, sub_assign);
impl_binary_op_inplace!(ops::MulAssign, mul_assign);
impl_binary_op_inplace!(ops::DivAssign, div_assign);
impl_binary_op_inplace!(ops::RemAssign, rem_assign);
impl_binary_op_inplace!(ops::BitAndAssign, bitand_assign);
impl_binary_op_inplace!(ops::BitOrAssign, bitor_assign);
impl_binary_op_inplace!(ops::BitXorAssign, bitxor_assign);
impl_binary_op_inplace!(ops::ShlAssign, shl_assign);
impl_binary_op_inplace!(ops::ShrAssign, shr_assign);

macro_rules! impl_unary_op {
    ($first:ident$(::$rest:ident)*, $fn:ident) => {
        // implement on reference
        impl<T> $first$(::$rest)* for TypedArray<T> 
        where T: PhysicalType + FromBytes + $first$(::$rest)*::<Output = T>,
        {
            type Output = TypedArray<T>;

            fn $fn(self) -> TypedArray<T> {
                self.elementwise_unary(|v| <T as $first$(::$rest)*>::$fn(v))
            }
        }
    };
}

impl_unary_op!(ops::Neg, neg);