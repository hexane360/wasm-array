use std::any::Any;

use zerocopy::{FromBytes, AsBytes};

use crate::dtype::{DataType, PhysicalType};
use crate::typedarray::{ArrRef, ArrRefMut, TypedArray};

pub struct Array {
    dtype: DataType,
    inner: Box<dyn Any>,
}

impl Array {
    pub fn dtype(&self) -> DataType { self.dtype }

    pub fn from_typed<T: PhysicalType + AsBytes + 'static>(arr: TypedArray<T>) -> Self {
        Self {
            dtype: T::DATATYPE,
            inner: Box::new(arr) as Box<dyn Any>,
        }
    }

    pub fn downcast<T: PhysicalType + AsBytes + 'static>(self) -> Option<TypedArray<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(*self.inner.downcast().unwrap())
    }

    pub fn downcast_ref<T: PhysicalType + AsBytes + 'static>(&self) -> Option<&TypedArray<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(self.inner.downcast_ref().unwrap())
    }

    pub fn downcast_mut<T: PhysicalType + AsBytes + 'static>(&mut self) -> Option<&mut TypedArray<T>> {
        if T::DATATYPE != self.dtype { return None; }
        Some(self.inner.downcast_mut().unwrap())
    }

    pub fn add_assign(&mut self, other: &Array) {
        assert!(self.dtype == other.dtype);

        match self.dtype() {
            DataType::UInt8 => self.downcast_mut::<u8>().unwrap().elementwise_binary_inplace(&other.downcast_ref::<u8>().unwrap(), |a, b| *a += *b),
            _ => panic!("Unsupported operation"),
        }
    }
}

macro_rules! unary_dispatch {
    ( $fn:expr, $( $ty:ident ),* ) => {
        match self.dtype() {
            $(
                <$ty as PhysicalType>::DATATYPE => self.downcast::<$ty>().unwrap().elementwise_unary($fn).type_erase(),
            )*
            _ => panic!("Unsupported operation for type '{}'", self.dtype()),
        }
    };
}

macro_rules! unary_dispatch_inplace {
    ( $fn:expr, $( $ty:ident ),* ) => {
        match self.dtype() {
            $(
                <$ty as PhysicalType>::DATATYPE => self.downcast_mut::<$ty>().unwrap().elementwise_unary_inplace($fn),
            )*
            _ => panic!("Unsupported operation for type '{}'", self.dtype()),
        }
    };
}