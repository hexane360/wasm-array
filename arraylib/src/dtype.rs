pub use num::complex::Complex;
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum DataTypeCategory {
    Boolean,
    Unsigned,
    Signed,
    Floating,
    Complex,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataTypeOrCategory {
    DataType(DataType),
    Category(DataTypeCategory)
}

mod private {
    use super::*;

    pub trait IntoDataTypePrivate {
        fn as_dtype(&self) -> super::DataType;

        fn as_dtype_categorized(&self) -> CategorizedDataType {
            self.as_dtype().as_dtype_categorized()
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum BooleanDataType {
        Boolean,
    }

    impl BooleanDataType {
        pub(crate) fn uncategorized(&self) -> DataType {
            match *self {
                BooleanDataType::Boolean => DataType::Boolean,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum UnsignedDataType {
        Generic,
        UInt8, UInt16, UInt32, UInt64,
    }

    impl UnsignedDataType {
        pub(crate) fn uncategorized(&self) -> DataType {
            match *self {
                UnsignedDataType::UInt8 => DataType::UInt8,
                UnsignedDataType::UInt16 => DataType::UInt16,
                UnsignedDataType::UInt32 => DataType::UInt32,
                UnsignedDataType::UInt64 | UnsignedDataType::Generic => DataType::UInt64,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum SignedDataType {
        Generic,
        Int8, Int16, Int32, Int64,
    }

    impl SignedDataType {
        pub(crate) fn uncategorized(&self) -> DataType {
            match *self {
                SignedDataType::Int8 => DataType::Int8,
                SignedDataType::Int16 => DataType::Int16,
                SignedDataType::Int32 => DataType::Int32,
                SignedDataType::Int64 | SignedDataType::Generic => DataType::Int64,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum FloatingDataType {
        Generic,
        Float32, Float64
    }

    impl FloatingDataType {
        pub(crate) fn uncategorized(&self) -> DataType {
            match *self {
                FloatingDataType::Float32 => DataType::Float32,
                FloatingDataType::Float64 | FloatingDataType::Generic => DataType::Float64,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum ComplexDataType {
        Generic,
        Complex64, Complex128
    }

    impl ComplexDataType {
        pub(crate) fn uncategorized(&self) -> DataType {
            match *self {
                ComplexDataType::Complex64 => DataType::Complex64,
                ComplexDataType::Complex128 | ComplexDataType::Generic => DataType::Complex128,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    pub enum CategorizedDataType {
        Boolean(BooleanDataType),
        Unsigned(UnsignedDataType),
        Signed(SignedDataType),
        Floating(FloatingDataType),
        Complex(ComplexDataType),
    }

    impl CategorizedDataType {
        pub(crate) fn category(&self) -> DataTypeCategory {
            match *self {
                CategorizedDataType::Boolean(_) => DataTypeCategory::Boolean,
                CategorizedDataType::Unsigned(_) => DataTypeCategory::Unsigned,
                CategorizedDataType::Signed(_) => DataTypeCategory::Signed,
                CategorizedDataType::Floating(_) => DataTypeCategory::Floating,
                CategorizedDataType::Complex(_) => DataTypeCategory::Complex,
            }
        }

        pub(crate) fn uncategorized(&self) -> DataType {
            match *self {
                CategorizedDataType::Boolean(inner) => inner.uncategorized(),
                CategorizedDataType::Unsigned(inner) => inner.uncategorized(),
                CategorizedDataType::Signed(inner) => inner.uncategorized(),
                CategorizedDataType::Floating(inner) => inner.uncategorized(),
                CategorizedDataType::Complex(inner) => inner.uncategorized(),
            }
        }

        pub(crate) fn promote_to_category(self, category: DataTypeCategory) -> Option<CategorizedDataType> {
            match (category, self) {
                (DataTypeCategory::Boolean, CategorizedDataType::Boolean(_)) => Some(self),
                (DataTypeCategory::Boolean, _) => None,

                (DataTypeCategory::Unsigned, CategorizedDataType::Boolean(_)) => Some(CategorizedDataType::Unsigned(UnsignedDataType::UInt8)),
                (DataTypeCategory::Unsigned, CategorizedDataType::Unsigned(_)) => Some(self),
                (DataTypeCategory::Unsigned, _) => None,

                (DataTypeCategory::Signed, CategorizedDataType::Boolean(_)) => Some(CategorizedDataType::Signed(SignedDataType::Int8)),
                (DataTypeCategory::Signed, CategorizedDataType::Unsigned(u)) => Some(CategorizedDataType::Signed(match u {
                    UnsignedDataType::Generic => SignedDataType::Generic,
                    UnsignedDataType::UInt8 => SignedDataType::Int16,
                    UnsignedDataType::UInt16 => SignedDataType::Int32,
                    UnsignedDataType::UInt32 | UnsignedDataType::UInt64 => SignedDataType::Int64,
                })),
                (DataTypeCategory::Signed, CategorizedDataType::Signed(_)) => Some(self),
                (DataTypeCategory::Signed, _) => None,

                (DataTypeCategory::Floating, CategorizedDataType::Boolean(_)) => Some(CategorizedDataType::Floating(FloatingDataType::Float32)),
                (DataTypeCategory::Floating, CategorizedDataType::Unsigned(u)) => Some(CategorizedDataType::Floating(match u {
                    UnsignedDataType::Generic => FloatingDataType::Generic,
                    UnsignedDataType::UInt8 | UnsignedDataType::UInt16 => FloatingDataType::Float32,
                    UnsignedDataType::UInt32 | UnsignedDataType::UInt64 => FloatingDataType::Float64,
                })),
                (DataTypeCategory::Floating, CategorizedDataType::Signed(s)) => Some(CategorizedDataType::Floating(match s {
                    SignedDataType::Generic => FloatingDataType::Generic,
                    SignedDataType::Int8 | SignedDataType::Int16 => FloatingDataType::Float32,
                    SignedDataType::Int32 | SignedDataType::Int64 => FloatingDataType::Float64,
                })),
                (DataTypeCategory::Floating, CategorizedDataType::Floating(_)) => Some(self),
                (DataTypeCategory::Floating, _) => None,

                (DataTypeCategory::Complex, CategorizedDataType::Boolean(_)) => Some(CategorizedDataType::Complex(ComplexDataType::Complex64)),
                (DataTypeCategory::Complex, CategorizedDataType::Unsigned(u)) => Some(CategorizedDataType::Complex(match u {
                    UnsignedDataType::Generic => ComplexDataType::Generic,
                    UnsignedDataType::UInt8 | UnsignedDataType::UInt16 => ComplexDataType::Complex64,
                    UnsignedDataType::UInt32 | UnsignedDataType::UInt64 => ComplexDataType::Complex128,
                })),
                (DataTypeCategory::Complex, CategorizedDataType::Signed(s)) => Some(CategorizedDataType::Complex(match s {
                    SignedDataType::Generic => ComplexDataType::Generic,
                    SignedDataType::Int8 | SignedDataType::Int16 => ComplexDataType::Complex64,
                    SignedDataType::Int32 | SignedDataType::Int64 => ComplexDataType::Complex128,
                })),
                (DataTypeCategory::Complex, CategorizedDataType::Floating(f)) => Some(CategorizedDataType::Complex(match f {
                    FloatingDataType::Generic => ComplexDataType::Generic,
                    FloatingDataType::Float32 => ComplexDataType::Complex64,
                    FloatingDataType::Float64 => ComplexDataType::Complex128,
                })),
                (DataTypeCategory::Complex, CategorizedDataType::Complex(_)) => Some(self),
            }
        }
    }
}

use private::{CategorizedDataType, BooleanDataType, UnsignedDataType, SignedDataType, FloatingDataType, ComplexDataType};

pub trait IntoDataType: private::IntoDataTypePrivate {
    fn as_dtype(&self) -> DataType { <Self as private::IntoDataTypePrivate>::as_dtype(self) }
}

impl<T> IntoDataType for T where T: private::IntoDataTypePrivate { }

impl private::IntoDataTypePrivate for DataType {
    fn as_dtype(&self) -> DataType { *self }

    fn as_dtype_categorized(&self) -> CategorizedDataType {
        match self {
            DataType::Boolean => CategorizedDataType::Boolean(BooleanDataType::Boolean),
            DataType::UInt8 => CategorizedDataType::Unsigned(UnsignedDataType::UInt8),
            DataType::UInt16 => CategorizedDataType::Unsigned(UnsignedDataType::UInt16),
            DataType::UInt32 => CategorizedDataType::Unsigned(UnsignedDataType::UInt32),
            DataType::UInt64 => CategorizedDataType::Unsigned(UnsignedDataType::UInt64),
            DataType::Int8 => CategorizedDataType::Signed(SignedDataType::Int8),
            DataType::Int16 => CategorizedDataType::Signed(SignedDataType::Int16),
            DataType::Int32 => CategorizedDataType::Signed(SignedDataType::Int32),
            DataType::Int64 => CategorizedDataType::Signed(SignedDataType::Int64),
            DataType::Float32 => CategorizedDataType::Floating(FloatingDataType::Float32),
            DataType::Float64 => CategorizedDataType::Floating(FloatingDataType::Float64),
            DataType::Complex64 => CategorizedDataType::Complex(ComplexDataType::Complex64),
            DataType::Complex128 => CategorizedDataType::Complex(ComplexDataType::Complex128),
        }
    }
}

impl private::IntoDataTypePrivate for DataTypeOrCategory {
    fn as_dtype(&self) -> DataType {
        match self {
            DataTypeOrCategory::DataType(dtype) => *dtype,
            DataTypeOrCategory::Category(category) => match category {
                DataTypeCategory::Boolean => DataType::Boolean,
                DataTypeCategory::Unsigned => DataType::UInt64,
                DataTypeCategory::Signed => DataType::Int64,
                DataTypeCategory::Floating => DataType::Float64,
                DataTypeCategory::Complex => DataType::Complex128,
            }
        }
    }

    fn as_dtype_categorized(&self) -> CategorizedDataType {
        match self {
            DataTypeOrCategory::DataType(dtype) => dtype.as_dtype_categorized(),
            DataTypeOrCategory::Category(category) => match category {
                DataTypeCategory::Boolean => CategorizedDataType::Boolean(BooleanDataType::Boolean),
                DataTypeCategory::Unsigned => CategorizedDataType::Unsigned(UnsignedDataType::Generic),
                DataTypeCategory::Signed => CategorizedDataType::Signed(SignedDataType::Generic),
                DataTypeCategory::Floating => CategorizedDataType::Floating(FloatingDataType::Generic),
                DataTypeCategory::Complex => CategorizedDataType::Complex(ComplexDataType::Generic),
            }
        }
    }
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

impl Into<&'static str> for DataType {
    fn into(self) -> &'static str {
        match self {
            DataType::Boolean => "bool",
            DataType::UInt8 => "uint8",
            DataType::UInt16 => "uint16",
            DataType::UInt32 => "uint32",
            DataType::UInt64 => "uint64",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
            DataType::Complex64 => "complex64",
            DataType::Complex128 => "complex128",
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.pad((*self).into())
    }
}

/// Perform type promotion on `tys`.
/// Promotion happens in two steps. First, we promote to the
/// common general `category` of types, e.g. unsigned, floating, complex.
/// Then, we promote as necessary within that category to prevent the loss of
/// information.
/// This roughly follows numpy's promotion convention, with a few differences.
/// Type promotions are commutative, but not necessarily associative. For instance,
/// `promote_type(&[Float32, promote_type(&[Int16, UInt16])]) == Float64`,
/// while `promote_type(&[Int16, promote_type(&[Float32, UInt16])]) == Float32`.
pub fn promote_types<'a, T, I>(dtypes: I) -> DataType
where T: IntoDataType + 'a, I: IntoIterator<Item = &'a T> + Copy {
    // we first find the output category as the maximum
    // of the input categories
    let min_category = match dtypes.into_iter().map(|s| s.as_dtype_categorized().category()).max() {
        Some(s) => s,
        // if no dtypes are passed, return float64
        None => return DataType::Float64,
    };

    // next, we promote each type to that category
    dtypes.into_iter().map(|s| s.as_dtype_categorized().promote_to_category(min_category).expect("Mismatch between min_category and promote_to_category")) 
    // and take the maximum within that category
        .max().unwrap().uncategorized()
}