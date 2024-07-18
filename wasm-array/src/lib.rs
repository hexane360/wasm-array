#![feature(iterator_try_collect)]
#![feature(try_blocks)]

extern crate alloc;

use std::collections::HashMap;
use std::panic::{catch_unwind, UnwindSafe};
use std::sync::OnceLock;

use arraylib::bool::Bool;
use num::Complex;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use ndarray::Array1;

use arraylib::array::DynArray;
use arraylib::dtype::DataType;
use arraylib::error::ArrayError;
use arraylib::{fft, reductions};

pub mod expr;
pub mod types;

use expr::{parse_with_literals, ArrayFunc, FuncMap, Token, UnaryFunc, BinaryFunc};
use types::{ArrayInterchange, parse_arraylike};

// # typescript exports

#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_PRELUDE: &'static str = r##"

type TypeStrEndianness = "<" | ">" | "|";
type TypeStrCharCode = "b" | "i" | "u" | "c" | "f";
type TypeStrItemSize = "1" | "2" | "4" | "8" | "16";

/**
 * Typestring in numpy's __array_interface__ protocol.
 * 
 * Not all datatypes are supported
 * 
 * Example: '<u4'
 */
type TypeStr = `${TypeStrEndianness}${TypeStrCharCode}${TypeStrItemSize}`

/**
 * Array interchange type.
 * 
 * Corresponds with numpy's __array_interface__ protocol.
 */
interface IArrayInterchange {
    data: BufferSource;
    typestr: TypeStr;
    shape: ShapeLike;
    strides: StridesLike;
    version: number;
}

/**
 * Color-like. Can be a string `"#rrggbb"` or `"#rrggbbaa"`, or an array of 3 or 4 floats.
 */
type ColorLike = string | readonly [number, number, number, number?];

type IntWidth = "8" | "16" | "32" | "64";
type FloatWidth = "32" | "64";
type ComplexWidth = "64" | "128";
/**
 * Datatype-like. Can be a string like "uint8", or a DataType object.
 */
type DataTypeLike = DataType | "bool" | `uint${IntWidth}` | `int${IntWidth}` | `float${FloatWidth}` | `complex${ComplexWidth}`;

type NestedArray<T> = T | ReadonlyArray<NestedArray<T>>;

/**
 * Array-like. Can be a number, `BigInt`, boolean, typed array, or nested array.
 */
type ArrayLike = NArray | NestedArray<number | BigInt | boolean | NArray> | ArrayBufferView;

type ShapeLike = ReadonlyArray<number> | Uint8Array | Uint16Array | Uint32Array | BigUint64Array;
type StridesLike = ShapeLike | Int8Array | Int16Array | Int32Array | BigInt64Array;
type AxesLike = StridesLike;

type FFTNorm = "backward" | "ortho" | "forward";

/**
 * Evaluate an arithemetic array expression. Designed to be used as a template literal.
 * 
 * Example: ```
 *   let result = expr`sqrt(${arr1}^2 + ${arr2}^2)`;
 * ```
 */
export function expr(strs: ReadonlyArray<string>, ...lits: ReadonlyArray<ArrayLike>): NArray;

/**
 * Return arrays representing the indices of a grid.
 * 
 * Returns an array for each of the dimensions in `shape`.
 * 
 * If `sparse` is `true`, return arrays of shape `(1, 1, ..., shape[i], ..., 1)`, such that they are broadcastable together.
 * Otherwise, all arrays have shape `shape`. `sparse` defaults to `false`.
 */
export function indices(shape: ShapeLike, dtype?: DataTypeLike, sparse?: boolean): Array<NArray>;

/**
 * Return an array constructed from the given values, of the specified dtype.
 */
export function array(arr: ArrayLike, dtype?: DataTypeLike): NArray;

/**
 * Return an array of integer indices with integral dtype `dtype`.
 */
export function arange(start: number, end?: number, dtype?: DataTypeLike): NArray;

/**
 * Return an array of `n` evenly spaced values with dtype `dtype`.
 * 
 * Includes both endpoints `[start, end]`.
 */
export function linspace(start: number, end: number, n: number, dtype?: DataTypeLike): NArray;

/**
 * Return an array of `n` values spaced on a logarithmic grid.
 * 
 * Values are logarithmically spaced in the interval `[base**start, base**end]`, with dtype `dtype`.
 */
export function logspace(start: number, end: number, n: number, dtype?: DataTypeLike, base?: number): NArray;

/**
 * Return an array of `n` values spaced geometrically.
 * 
 * Values are logarithmically spaced in the interval `[start, end]`, with dtype `dtype`.
 */
export function geomspace(start: number, end: number, n: number, dtype?: DataTypeLike): NArray;

/**
 * Construct an identity matrix of `ndim` dimensions, with dtype `dtype`.
 */
export function eye(ndim: number, dtype?: DataTypeLike): NArray;
"##;

// # wasm imports

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(typescript_type = "IArrayInterchange")]
    pub type IArrayInterchange;

    #[wasm_bindgen(typescript_type = "ArrayLike")]
    pub type ArrayLike;

    #[wasm_bindgen(typescript_type = "DataTypeLike")]
    pub type DataTypeLike;

    #[wasm_bindgen(typescript_type = "ColorLike")]
    pub type ColorLike;

    #[wasm_bindgen(typescript_type = "ShapeLike")]
    pub type ShapeLike;

    #[wasm_bindgen(typescript_type = "AxesLike")]
    pub type AxesLike;

    #[wasm_bindgen(typescript_type = "FFTNorm")]
    pub type FFTNorm;
}

// # struct definitions

#[wasm_bindgen(js_name = DataType)]
#[derive(Clone, Copy)]
pub struct JsDataType {
    inner: DataType,
}

impl From<DataType> for JsDataType {
    fn from(value: DataType) -> Self { JsDataType { inner: value } }
}

#[wasm_bindgen(js_class = DataType)]
impl JsDataType {
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        self.inner.to_string()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("DataType(\"{}\")", self.inner.to_string())
    }
}

#[wasm_bindgen(js_name = NArray)]
pub struct JsArray {
    inner: DynArray
}

impl From<DynArray> for JsArray {
    fn from(value: DynArray) -> Self { JsArray { inner: value } }
}


#[wasm_bindgen(js_class = NArray)]
impl JsArray {
    #[wasm_bindgen(getter)]
    /// Return the dtype of the array.
    pub fn dtype(&self) -> JsDataType {
        self.inner.dtype().into()
    }

    #[wasm_bindgen(getter)]
    /// Return the shape of the array.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    #[wasm_bindgen(getter)]
    /// Return the size (total number of elements) of the array
    pub fn size(&self) -> usize {
        self.inner.shape().iter().product()
    }

    #[wasm_bindgen(getter)]
    /// Return the itemsize, in bytes, of each array element
    pub fn itemsize(&self) -> usize {
        self.inner.dtype().item_size()
    }

    #[wasm_bindgen(js_name = toJSON)]
    /// Convert the array to a simple JSON representation.
    pub fn to_json(&self) -> Result<js_sys::Object, JsValue> {
        let map = js_sys::Map::new();
        map.set(&"dtype".to_owned().into(), &self.inner.dtype().to_string().into());
        let shape = self.inner.shape();
        map.set(&"shape".to_owned().into(), &js_sys::Array::from_iter(shape.into_iter().map(|v| JsValue::from_f64(v as f64))));
        js_sys::Object::from_entries(&map.into())
    }

    #[wasm_bindgen(js_name = toInterchange)]
    /// Convert the array to a JSON interchange format, loosely conforming with numpy's __array_interface__ protocol.
    pub fn to_interchange(self) -> Result<JsValue, JsValue> {
        Ok(ArrayInterchange::from(self.inner).serialize(
            &serde_wasm_bindgen::Serializer::new()
                .serialize_missing_as_null(true)
                .serialize_maps_as_objects(true)
        ).map_err(|e| e.to_string())?)
    }

    #[wasm_bindgen(js_name = toString)]
    /// Convert the array to a string representation. Useful for debugging.
    pub fn to_string(&self) -> String {
        let mut s = format!("Array {}\n{}", self.inner.dtype(), self.inner);
        if s.lines().count() < 3 {
            let i = s.find('\n').unwrap();
            // SAFETY: we replace an entire UTF-8 codepoint '\n' with ' '
            unsafe { s.as_mut_vec()[i] = b' ' }
        }
        s
    }

    /// Broadcast the array to shape `shape`. Throws an error if the broadcast is not possible.
    pub fn broadcast_to(self, shape: ShapeLike) -> Result<JsArray, String> {
        let shape: Box<[usize]> = shape.try_into()?;
        match self.inner.broadcast_to(&*shape) {
            Ok(v) => Ok(v.into()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Apply a colormap to the array, assuming values are normalized between 0 and 1.
    pub fn apply_cmap(&self, min_color: Option<ColorLike>, max_color: Option<ColorLike>, invalid_color: Option<ColorLike>) -> Result<Clamped<Vec<u8>>, String> {
        let min_color: Option<Array1<f32>> = min_color.map(|c| c.try_into()).transpose()?;
        //let min_color = if min_color.is_null() { None } else { Some(get_color(min_color)?) };
        //let max_color = if max_color.is_null() { None } else { Some(get_color(max_color)?) };
        //let invalid_color = if invalid_color.is_null() { None } else { Some(get_color(invalid_color)?) };

        Ok(Clamped(self.inner.apply_cmap().downcast::<u8>().unwrap().into_raw_vec()))
    }

    /// Return the array converted to datatype `dtype`. Throws an error if the conversion is not possible.
    pub fn astype(&self, dtype: &DataTypeLike) -> Result<JsArray, String> {
        catch_panic(|| {
            Ok(self.inner.cast(dtype.try_into()?).into_owned().into())
        })
    }

    /// Reshape array into shape `shape`.
    /// 
    /// Up to one axis can be specified as '-1', allowing it to be inferred from the length of the array.
    pub fn reshape(&self, shape: AxesLike) -> Result<JsArray, String> {
        catch_panic(|| {
            let shape: Box<[isize]> = serde_wasm_bindgen::from_value(shape.obj).map_err(|e| e.to_string())?;

            Ok(self.inner.reshape(&shape)?.into())
        })
    }

    #[wasm_bindgen]
    /// Return a contiguous, flattened array.
    pub fn ravel(&self) -> Result<JsArray, String> {
        catch_panic(|| {
            Ok(self.inner.ravel().into())
        })
    }

    #[wasm_bindgen]
    /// Return a contiguous, flattened array.
    pub fn flatten(&self) -> Result<JsArray, String> {
        catch_panic(|| {
            Ok(self.inner.ravel().into())
        })
    }
}

// # array functions
// ## construction functions

#[wasm_bindgen]
/// Create a `DataType` object from a `DataTypeLike`.
pub fn to_dtype(dtype: &DataTypeLike) -> Result<JsDataType, String> {
    Ok(<&DataTypeLike as TryInto<DataType>>::try_into(dtype)?.into())
}

#[wasm_bindgen(skip_typescript)]
pub fn array(arr: &ArrayLike, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            None
        } else { Some(dtype.try_into()?) };

        parse_arraylike(arr, dtype).map(|val| val.into_owned().into())
    })
}

#[wasm_bindgen]
/// Return an array filled with ones, of shape `shape` and dtype `dtype`
pub fn ones(shape: ShapeLike, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let shape: Box<[usize]> = shape.try_into()?;
        let dtype = dtype.try_into()?;
        Ok(DynArray::ones(shape.as_ref(), dtype).into())
    })
}

#[wasm_bindgen]
/// Return an array filled with zeros, of shape `shape` and dtype `dtype`
pub fn zeros(shape: ShapeLike, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let shape: Box<[usize]> = shape.try_into()?;
        let dtype = dtype.try_into()?;
        Ok(DynArray::zeros(shape.as_ref(), dtype).into())
    })
}

#[wasm_bindgen(skip_typescript)]
pub fn arange(start: f64, end: Option<f64>, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            DataType::Int64
        } else { dtype.try_into()? };

        let (start, end) = match (start, end) {
            (start, Some(end)) => (start, end),
            (start, None) => (0., start),
        };

        Ok(match dtype {
            DataType::UInt8 => DynArray::arange(start as u8, end as u8),
            DataType::UInt16 => DynArray::arange(start as u16, end as u16),
            DataType::UInt32 => DynArray::arange(start as u32, end as u32),
            DataType::UInt64 => DynArray::arange(start as u64, end as u64),
            DataType::Int8 => DynArray::arange(start as i8, end as i8),
            DataType::Int16 => DynArray::arange(start as i16, end as i16),
            DataType::Int32 => DynArray::arange(start as i32, end as i32),
            DataType::Int64 => DynArray::arange(start as i64, end as i64),
            dtype => return Err(format!("'arange' not supported for dtype '{}'", dtype)),
        }.into())
    })
}

#[wasm_bindgen(skip_typescript)]
pub fn indices(shape: ShapeLike, dtype: &DataTypeLike, sparse: Option<bool>) -> Result<Vec<JsArray>, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            DataType::Int64
        } else { dtype.try_into()? };
        let shape: Box<[usize]> = shape.try_into()?;
        let sparse = sparse.unwrap_or(false);

        Ok(match dtype {
            DataType::UInt8 => DynArray::indices::<u8>(&shape, sparse),
            DataType::UInt16 => DynArray::indices::<u16>(&shape, sparse),
            DataType::UInt32 => DynArray::indices::<u32>(&shape, sparse),
            DataType::UInt64 => DynArray::indices::<u64>(&shape, sparse),
            DataType::Int8 => DynArray::indices::<i8>(&shape, sparse),
            DataType::Int16 => DynArray::indices::<i16>(&shape, sparse),
            DataType::Int32 => DynArray::indices::<i32>(&shape, sparse),
            DataType::Int64 => DynArray::indices::<i64>(&shape, sparse),
            dtype => return Err(format!("'arange' not supported for dtype '{}'", dtype)),
        }.into_iter().map(|arr| arr.into()).collect())
    })
}

#[wasm_bindgen(skip_typescript)]
pub fn linspace(start: f64, end: f64, n: usize, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            DataType::Float64
        } else { dtype.try_into()? };

        Ok(match dtype {
            DataType::Float32 => DynArray::linspace(start as f32, end as f32, n),
            DataType::Float64 => DynArray::linspace(start as f64, end as f64, n),
            dtype => return Err(format!("'linspace' not supported for dtype '{}'", dtype)),
        }.into())
    })
}

#[wasm_bindgen(skip_typescript)]
pub fn logspace(start: f64, end: f64, n: usize, dtype: &DataTypeLike, base: Option<f64>) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            DataType::Float64
        } else { dtype.try_into()? };
        let base = base.unwrap_or(10.);

        Ok(match dtype {
            DataType::Float32 => DynArray::logspace(start as f32, end as f32, n, base as f32),
            DataType::Float64 => DynArray::logspace(start as f64, end as f64, n, base),
            dtype => return Err(format!("'logspace' not supported for dtype '{}'", dtype)),
        }.into())
    })
}

#[wasm_bindgen(skip_typescript)]
pub fn geomspace(start: f64, end: f64, n: usize, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            DataType::Float64
        } else { dtype.try_into()? };

        Ok(match dtype {
            DataType::Float32 => DynArray::geomspace(start as f32, end as f32, n),
            DataType::Float64 => DynArray::geomspace(start as f64, end as f64, n),
            dtype => return Err(format!("'geomspace' not supported for dtype '{}'", dtype)),
        }.into())
    })
}

#[wasm_bindgen(skip_typescript)]
pub fn eye(ndim: f64, dtype: &DataTypeLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
            DataType::Float64
        } else { dtype.try_into()? };
        let ndim = ndim as usize;

        Ok(match dtype {
            DataType::Boolean => DynArray::eye::<Bool>(ndim),
            DataType::UInt8 => DynArray::eye::<u8>(ndim),
            DataType::UInt16 => DynArray::eye::<u16>(ndim),
            DataType::UInt32 => DynArray::eye::<u32>(ndim),
            DataType::UInt64 => DynArray::eye::<u64>(ndim),
            DataType::Int8 => DynArray::eye::<i8>(ndim),
            DataType::Int16 => DynArray::eye::<i16>(ndim),
            DataType::Int32 => DynArray::eye::<i32>(ndim),
            DataType::Int64 => DynArray::eye::<i64>(ndim),
            DataType::Float32 => DynArray::eye::<f32>(ndim),
            DataType::Float64 => DynArray::eye::<f64>(ndim),
            DataType::Complex64 => DynArray::eye::<Complex<f32>>(ndim),
            DataType::Complex128 => DynArray::eye::<Complex<f64>>(ndim),
        }.into())
    })
}

// ## reshaping functions

#[wasm_bindgen]
/// Roll elements of the array by `shifts`, along axes `axes`.
///
/// If `axes` is specified, `shifts` and `axes` must be the same length.
/// Otherwise, `shifts` must be the same length as the array's dimensionality.
pub fn roll(arr: &ArrayLike, shifts: AxesLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    // TODO support single value shift arguments
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let shifts: Box<[isize]> = serde_wasm_bindgen::from_value(shifts.obj).map_err(|e| e.to_string())?;

        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };

        if shifts.len() != axes.len() {
            return Err(format!("'shifts' must be the same length as 'axes' (or the array's ndim, if 'axes' is not specified)."))
        }

        Ok(arr.as_ref().roll(&shifts, &axes).into())
    })
}

#[wasm_bindgen]
/// Reshape array into shape `shape`.
/// 
/// Up to one axis can be specified as '-1', allowing it to be inferred from the length of the array.
pub fn reshape(arr: &ArrayLike, shape: AxesLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let shape: Box<[isize]> = serde_wasm_bindgen::from_value(shape.obj).map_err(|e| e.to_string())?;

        Ok(arr.as_ref().reshape(&shape)?.into())
    })
}

#[wasm_bindgen]
/// Return a contiguous, flattened array.
pub fn ravel(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(arr.as_ref().ravel().into())
    })
}

// ## elementwise functions

#[wasm_bindgen]
/// Return the ceiling of the input, element-wise
pub fn ceil(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(arr.as_ref().ceil().into())
    })
}

#[wasm_bindgen]
/// Return the floor of the input, element-wise
pub fn floor(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(arr.as_ref().ceil().into())
    })
}

#[wasm_bindgen]
/// Return the absolute value of the input, element-wise
pub fn abs(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(arr.as_ref().abs().into())
    })
}

#[wasm_bindgen]
/// Return the complex conjugate of the input, element-wise
pub fn conj(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?; 
        Ok(arr.as_ref().conj().into())
    })
}

#[wasm_bindgen]
/// Return the square root of the input, element-wise
pub fn sqrt(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(arr.as_ref().sqrt().into())
    })
}

#[wasm_bindgen]
/// Return the smallest element between the two arrays, elementwise.
/// 
/// Propagates NaNs, preferring the first value if both are NaN.
pub fn minimum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr1 = parse_arraylike(arr1, None)?;
        let arr2 = parse_arraylike(arr2, None)?;
        Ok(arr1.as_ref().minimum(arr2.as_ref()).into())
    })
}

#[wasm_bindgen]
/// Return the largest element between the two arrays, elementwise.
/// 
/// Propagates NaNs, preferring the first value if both are NaN.
pub fn maximum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr1 = parse_arraylike(arr1, None)?;
        let arr2 = parse_arraylike(arr2, None)?;
        Ok(arr1.as_ref().maximum(arr2.as_ref()).into())
    })
}

#[wasm_bindgen]
/// Return the smallest element between the two arrays, elementwise.
/// 
/// Ignores NaNs. If both values are NaN, returns the first one.
pub fn nanminimum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr1 = parse_arraylike(arr1, None)?;
        let arr2 = parse_arraylike(arr2, None)?;
        Ok(arr1.as_ref().nanminimum(arr2.as_ref()).into())
    })
}

#[wasm_bindgen]
/// Return the largest element between the two arrays, elementwise.
///
/// Ignores NaNs. If both values are NaN, returns the first one.
pub fn nanmaximum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr1 = parse_arraylike(arr1, None)?;
        let arr2 = parse_arraylike(arr2, None)?;
        Ok(arr1.as_ref().nanmaximum(arr2.as_ref()).into())
    })
}

// ## reductions

#[wasm_bindgen]
// Return the minimum element along the given axes.
//
// NaN values are propagated. See `nanmin` for a version that ignores missing values.
pub fn min(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::min(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the maximum element along the given axes.
//
// NaN values are propagated. See `nanmax` for a version that ignores missing values.
pub fn max(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::max(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the sum of elements along the given axes.
//
// NaN values are propagated. See `nansum` for a version that ignores missing values.
pub fn sum(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::sum(arr.as_ref(), axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the product of elements along the given axes.
//
// NaN values are propagated. See `nanprod` for a version that ignores missing values.
pub fn prod(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::prod(arr.as_ref(), axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the mean element along the given axes.
//
// NaN values are propagated. See `nanmean` for a version that ignores missing values.
pub fn mean(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::mean(arr.as_ref(), axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the minimum element along the given axes.
//
// NaN values are ignored.
pub fn nanmin(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::nanmin(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the maximum element along the given axes.
//
// NaN values are ignored.
pub fn nanmax(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::nanmax(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the sum of elements along the given axes.
//
// NaN values are ignored.
pub fn nansum(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::nansum(arr.as_ref(), axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the product of elements along the given axes.
//
// NaN values are ignored.
pub fn nanprod(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::nanprod(arr.as_ref(), axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the mean element along the given axes.
//
// NaN values are ignored.
pub fn nanmean(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::nanmean(arr.as_ref(), axes.as_deref()).into())
    })
}

// ## FFT functions

#[wasm_bindgen]
/// Compute the Fourier transform of the input array
/// 
/// Computes the transformation along each of `axes` (defaults to all axes).
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn fft(arr: &ArrayLike, axes: Option<AxesLike>, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::fft(arr.as_ref(), &axes, norm).into())
    })
}

#[wasm_bindgen]
/// Compute the inverse Fourier transform of the input array
/// 
/// Computes the transformation along each of `axes` (defaults to all axes).
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn ifft(arr: &ArrayLike, axes: Option<AxesLike>, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::ifft(arr.as_ref(), &axes, norm).into())
    })
}

#[wasm_bindgen]
/// Shifts the zero-frequency component of a Fourier transformed array to the center
/// 
/// Shifts along each of `axes` (defaults to all axes).
pub fn fftshift(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };

        Ok(fft::fftshift(arr.as_ref(), &axes).into())
    })
}

#[wasm_bindgen]
/// Inverse of `fftshift`. Shifts the zero-frequency component of a Fourier transformed array to the corner
/// 
/// Shifts along each of `axes` (defaults to all axes).
pub fn ifftshift(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };

        Ok(fft::ifftshift(arr.as_ref(), &axes).into())
    })
}

#[wasm_bindgen]
/// Compute the Fourier transform of the input array
/// 
/// Computes the transformation along the last two axes of the input.
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn fft2(arr: &ArrayLike, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::fft(arr.as_ref(), &[-2, -1], norm).into())
    })
}

#[wasm_bindgen]
/// Compute the inverse Fourier transform of the input array
/// 
/// Computes the transformation along the last two axes of the input.
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn ifft2(arr: &ArrayLike, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::ifft(arr.as_ref(), &[-2, -1], norm).into())
    })
}

#[wasm_bindgen]
/// Shifts the zero-frequency component of a Fourier transformed array to the center
/// 
/// Computes the transformation along the last two axes of the input.
pub fn fft2shift(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(fft::fftshift(arr.as_ref(), &[-2, -1]).into())
    })
}

#[wasm_bindgen]
/// Inverse of `fft2shift`. Shifts the zero-frequency component of a Fourier transformed array to the corner
/// 
/// Computes the transformation along the last two axes of the input.
pub fn ifft2shift(arr: &ArrayLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let arr = parse_arraylike(arr, None)?;
        Ok(fft::ifftshift(arr.as_ref(), &[-2, -1]).into())
    })
}

// ## reductions to bool

#[wasm_bindgen]
pub fn allequal(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<bool, String> {
    catch_panic(|| {
        let arr1 = parse_arraylike(arr1, None)?;
        let arr2 = parse_arraylike(arr2, None)?;
        Ok(arr1.as_ref().allequal(arr2.as_ref()))
    })
}

#[wasm_bindgen]
pub fn allclose(arr1: &ArrayLike, arr2: &ArrayLike, rtol: Option<f64>, atol: Option<f64>) -> Result<bool, String> {
    catch_panic(|| {
        let arr1 = parse_arraylike(arr1, None)?;
        let arr2 = parse_arraylike(arr2, None)?;
        Ok(arr1.as_ref().allclose(arr2.as_ref(), rtol.unwrap_or(1e-8), atol.unwrap_or(0.0)))
    })
}

// ## from_interchange

#[wasm_bindgen]
/// Create an array from a JSON interchange format, loosely conforming with numpy's __array_interface__ protocol.
pub fn from_interchange(obj: IArrayInterchange) -> Result<JsArray, String> {
    catch_panic(|| {
        Ok(ArrayInterchange::try_from(obj)?.to_array()?.into())
    })
}

// ## constants
// TODO: make these properties

#[wasm_bindgen]
/// Return the constant pi
pub fn pi() -> JsArray {
    DynArray::from_val(std::f64::consts::PI).into()
}

#[wasm_bindgen]
/// Return the constant tau = 2pi
pub fn tau() -> JsArray {
    DynArray::from_val(std::f64::consts::TAU).into()
}

#[wasm_bindgen]
/// Return a NaN value
pub fn nan() -> JsArray {
    DynArray::from_val(std::f64::NAN).into()
}

#[wasm_bindgen]
/// Return an infinity value
pub fn inf() -> JsArray {
    DynArray::from_val(std::f64::INFINITY).into()
}

static ARRAY_FUNCS: OnceLock<FuncMap> = OnceLock::new();

fn init_array_funcs() -> FuncMap {
    let funcs: Vec<Box<dyn ArrayFunc + Sync + Send>> = vec![
        Box::new(UnaryFunc::new("abs", |v| v.abs())),
        Box::new(UnaryFunc::new("exp", |v| v.exp())),
        Box::new(UnaryFunc::new("sqrt", |v| v.sqrt())),
        Box::new(BinaryFunc::new("minimum", |l, r| l.minimum(r))),
        Box::new(BinaryFunc::new("maximum", |l, r| l.maximum(r))),
        Box::new(BinaryFunc::new("nanminimum", |l, r| l.nanminimum(r))),
        Box::new(BinaryFunc::new("nanmaximum", |l, r| l.nanmaximum(r))),
    ];

    funcs.into_iter().map(|f| (f.name(), f)).collect()
}

#[wasm_bindgen(variadic, skip_typescript)]
pub fn expr(strs: Vec<String>, lits: &JsValue) -> Result<JsArray, String> {
    catch_panic(|| {
        let lits = lits.clone().dyn_into::<js_sys::Array>().map_err(|_| "'lits' must be an array".to_owned())?;
        let lits: Vec<_> = lits.iter().map(|val| parse_arraylike(&val, None).map(|v| v.into_owned())).try_collect()?;

        let funcs = ARRAY_FUNCS.get_or_init(init_array_funcs);
        //return Err(format!("strs: {:?} lits: {:?}", strs, lits.into_iter().map(|a| a.inner).collect_vec()));
        let expr = parse_with_literals(strs.iter().map(|s| s.as_ref()), lits.into_iter().map(Token::ArrayLit))
            .map_err(|e| format!("{:?}", e))?;
        let vars = HashMap::new();
        match expr.exec(&vars, funcs) {
            Ok(arr) => Ok(arr.into()),
            Err(e) => Err(format!("{:?}", e)),
        }
    })
}

fn catch_panic<T, F: FnOnce() -> Result<T, String> + UnwindSafe>(f: F) -> Result<T, String> {
    match catch_unwind(f) {
        Ok(result) => result,
        Err(e) => Err(match e.downcast_ref::<ArrayError>() {
            Some(e) => e.to_string(),
            None => format!("panicked!")
        })
    }
}

pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    //set_panic_hook();
    Ok(())
}