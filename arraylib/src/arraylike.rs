
use std::iter;

use ndarray::ArrayD;

use crate::array::DynArray;
use crate::dtype::{Bool, Complex, DataType, DataTypeCategory, PhysicalType};
use crate::cast::CastFrom;

#[derive(Debug)]
pub enum NestedList {
    Value(ArrayValue),
    Array(Vec<NestedList>),
}

impl NestedList {
    pub fn infer_dtype(&self) -> DataType {
        let category = self.iter_values().map(|val| val.dtype_category())
            .max().unwrap_or(DataTypeCategory::Floating);

        match category {
            DataTypeCategory::Boolean => DataType::Boolean,
            DataTypeCategory::Unsigned => DataType::UInt64,
            DataTypeCategory::Signed => DataType::Int64,
            DataTypeCategory::Floating => DataType::Float64,
            DataTypeCategory::Complex => DataType::Complex128,
        }
    }

    pub fn iter_values<'a>(&'a self) -> Box<dyn Iterator<Item = ArrayValue> + 'a> {
        match self {
            NestedList::Value(val) => Box::new(iter::once(val.clone())),
            NestedList::Array(arr) => Box::new(arr.iter().flat_map(|list| list.iter_values())),
        }
    }

    fn get_shape(&self, dim: usize) -> Result<Vec<usize>, String> {
        match self {
            NestedList::Value(_) => Ok(vec![]),
            NestedList::Array(arr) => {
                let mut it = arr.iter().enumerate();
                let mut first_shape: Vec<usize> = match it.next() {
                    None => return Ok(vec![0]),
                    Some((_, inner)) => inner.get_shape(dim + 1)?,
                };
                for (i, inner) in it {
                    let shape = inner.get_shape(dim + 1)?;
                    if shape != first_shape {
                        return Err(format!(
                            "Shape mismatch at dim {}: Subarray 0 is of shape {:?}, but subarray {} is of shape {:?}",
                            dim, first_shape, i, shape
                        ));
                    }
                }
                first_shape.insert(0, arr.len());
                Ok(first_shape)
            }
        }
    }

    pub fn build_array(&self, datatype: Option<DataType>) -> Result<DynArray, String> {
        let shape = self.get_shape(0)?;

        let dtype = datatype.unwrap_or_else(|| self.infer_dtype());

        Ok(match dtype {
            DataType::Boolean => { let mut vec: Vec<Bool> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::UInt8 => { let mut vec: Vec<u8> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::UInt16 => { let mut vec: Vec<u16> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::UInt32 => { let mut vec: Vec<u32> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::UInt64 => { let mut vec: Vec<u64> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Int8 => { let mut vec: Vec<i8> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Int16 => { let mut vec: Vec<i16> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Int32 => { let mut vec: Vec<i32> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Int64 => { let mut vec: Vec<i64> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Float32 => { let mut vec: Vec<f32> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Float64 => { let mut vec: Vec<f64> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Complex64 => { let mut vec: Vec<Complex<f32>> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
            DataType::Complex128 => { let mut vec: Vec<Complex<f64>> = Vec::new(); let caster = ArrayValueCaster::make(); self.build_array_inner(&caster, &mut vec)?; ArrayD::from_shape_vec(shape, vec).unwrap().into() },
        })
    }

    fn build_array_inner<T: PhysicalType + CastFrom>(&self, caster: &ArrayValueCaster<T>, vec: &mut Vec<T>) -> Result<(), String> {
        match self {
            NestedList::Value(val) => Ok(vec.push(caster.cast(*val)
                .ok_or_else(|| format!("Unable to build array of dtype '{}' from value of type '{}'", T::DATATYPE, val.dtype_category()))?
            )),
            NestedList::Array(arr) => arr.iter().try_for_each(|val| val.build_array_inner(caster, vec)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArrayValue {
    Complex(Complex<f64>),
    Float(f64),
    Int(i64),
    Boolean(Bool),
}

impl ArrayValue {
    pub fn dtype_category(&self) -> DataTypeCategory {
        match self {
            ArrayValue::Complex(_) => DataTypeCategory::Complex,
            ArrayValue::Float(_) => DataTypeCategory::Floating,
            ArrayValue::Int(_) => DataTypeCategory::Signed,
            ArrayValue::Boolean(_) => DataTypeCategory::Boolean,
        }
    }
}

struct ArrayValueCaster<T> {
    complex_strategy: Option<fn(Complex<f64>) -> T>,
    float_strategy: Option<fn(f64) -> T>,
    int_strategy: Option<fn(i64) -> T>,
    bool_strategy: Option<fn(Bool) -> T>,
}

impl<T: PhysicalType + CastFrom> ArrayValueCaster<T> {
    pub fn make() -> Self { Self {
        complex_strategy: T::cast_from_complex128(),
        float_strategy: T::cast_from_float64(),
        int_strategy: T::cast_from_int64(),
        bool_strategy: T::cast_from_bool(),
    }}

    pub fn cast(&self, val: ArrayValue) -> Option<T> {
        Some(match val {
            ArrayValue::Complex(val) => self.complex_strategy?(val),
            ArrayValue::Float(val) => self.float_strategy?(val),
            ArrayValue::Int(val) => self.int_strategy?(val),
            ArrayValue::Boolean(val) => self.bool_strategy?(val),
        })
    }
}