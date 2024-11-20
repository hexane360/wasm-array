#![feature(iterator_try_collect)]
#![feature(try_blocks)]

extern crate alloc;

use std::collections::HashMap;
use std::sync::OnceLock;

use num::Complex;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use ndarray::{Array1, SliceInfoElem};

use arraylib::bool::Bool;
use arraylib::array::{DynArray, self};
use arraylib::colors::get_cmap;
use arraylib::dtype::DataType;
use arraylib::{fft, reductions};

pub mod expr;
pub mod types;

use expr::{parse_with_literals, ArrayFunc, FuncMap, Token, UnaryFunc, BinaryFunc};
use types::{ArrayInterchange, parse_index, parse_arraylike, to_nested_array};

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
    data: BufferSource | string | ReadonlyArray<number>;
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

type IndexLike = number | Slice | null;

type FFTNorm = "backward" | "ortho" | "forward";

declare module "." {
    interface NArray {
        slice(...idxs: ReadonlyArray<IndexLike>): NArray;
    }
}

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
 * Return a list of coordinate matrices from 1D coordinate vectors.
 * 
 * Given input vectors of lengths `(n_1, n_2, ... n_n)`, returns an
 * array of vectors of shape `(n_1, n_2, ... n_n)`, with values
 * corresponding to the values of the input arrays.
 */
export function meshgrid(...arr: ReadonlyArray<ArrayLike>): Array<NArray>;

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

/**
 * 1D linear interpolation.
 * Interpolates a set of values `xs` onto the piecewise line defined by `xp` and `yp`.
 * 
 * `xp` and `yp` must be 1D arrays. `xp` must be sorted and must not contain NaN values.
 * `left` and `right` define values to return in the case of `xs` outside `xp`. They default to the
 * first and last element of `yp` respectively.
 * 
 * Returns an array of the same shape as `xs`.
 */
export function interp(xs: ArrayLike, xp: ArrayLike, yp: ArrayLike, left?: number, right?: number): NArray;

/**
 * N-dimensional linear interpolation.
 * Interpolates a set of values `xs` onto the surface defined by coordinates `coords` and
 * values `values`.
 * 
 * `coords` must be a list of 1D arrays, sorted and without NaN values.
 * `values` must be an array of shape `[coords[0].size(), coords[1].size(), ..., coords[N-1].size()]`
 * `xs` is an array of points to interpolate at, with the last axis corresponding to the dimensions of `coords`.
 * 
 * Returns an array of shape `xs.shape[:-1]`.
 */
export function interpn(coords: ReadonlyArray<ArrayLike>, values: ArrayLike, xs: ArrayLike, fill?: number): NArray;

/**
 * Broadcast arrays together
 */
export function broadcast_arrays(...arrs: ReadonlyArray<ArrayLike>): Array<NArray>;

/**
 * Broadcast shapes together
 */
export function broadcast_shapes(...shapes: ReadonlyArray<ShapeLike>): Array<number>;
"##;

// # wasm imports

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name="log")]
    fn _log(s: &str);

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

    #[wasm_bindgen(typescript_type = "IndexLike")]
    pub type IndexLike;

    #[wasm_bindgen(typescript_type = "FFTNorm")]
    pub type FFTNorm;

    #[wasm_bindgen(typescript_type = "NestedArray<number | boolean>")]
    pub type NestedArray;
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

#[wasm_bindgen(js_name = Slice)]
#[derive(Clone, Copy)]
pub struct JsSlice {
    start: isize,
    end: Option<isize>,
    step: isize,
}

#[wasm_bindgen(js_class = Slice)]
impl JsSlice {
    #[wasm_bindgen(constructor)]
    pub fn new(start: Option<isize>, end: Option<isize>, step: Option<isize>) -> Self {
        let step = step.unwrap_or(1);
        return Self {
            start: start.unwrap_or(if step >= 0 { 0 } else { -1 }),
            end, step,
        }
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!(
            "{}..{}{}",
            if self.start == 0 { "".to_owned() } else { self.start.to_string() },
            if let Some(end) = self.end { end.to_string() } else { "".to_owned() },
            if self.step == 1 { "".to_owned() } else { format!("..{}", self.step) }
        )
    }
}

impl JsSlice {
    pub fn into_sliceinfoelem(&self) -> SliceInfoElem {
        SliceInfoElem::Slice {
            start: self.start,
            end: self.end,
            step: self.step,
        }
    }
}

#[wasm_bindgen(js_name = NArray)]
pub struct JsArray {
    inner: DynArray
}

impl From<DynArray> for JsArray {
    fn from(value: DynArray) -> Self {
        //_log(&format!("JsArray::new(dtype: {} shape: {:?})", value.dtype(), value.shape()));
        JsArray { inner: value }
    }
}

impl Into<DynArray> for JsArray {
    fn into(self) -> DynArray { self.inner }
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

    #[wasm_bindgen]
    pub fn clone(&self) -> JsArray {
        self.inner.clone().into()
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
        Ok(ArrayInterchange::from(Into::<DynArray>::into(self)).serialize(
            &serde_wasm_bindgen::Serializer::new()
                .serialize_missing_as_null(true)
                .serialize_maps_as_objects(true)
        ).map_err(|e| e.to_string())?)
    }

    #[wasm_bindgen(js_name = toString)]
    /// Convert the array to a string representation. Useful for debugging.
    pub fn to_string(&self) -> String {
        let mut s = format!("Array {}\n{}", self.inner.dtype(), &self.inner);
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
    pub fn apply_cmap(&self, cmap: &str, min_color: Option<ColorLike>, max_color: Option<ColorLike>, invalid_color: Option<ColorLike>) -> Result<Clamped<Vec<u8>>, String> {
        let cmap = get_cmap(cmap)?;
        let min_color: Option<Array1<f32>> = min_color.map(|c| c.try_into()).transpose()?;
        let max_color: Option<Array1<f32>> = max_color.map(|c| c.try_into()).transpose()?;
        let invalid_color: Array1<f32> = invalid_color.map(|c| c.try_into()).transpose()?.unwrap_or(Array1::<f32>::zeros((4,)));
        //let min_color = if min_color.is_null() { None } else { Some(get_color(min_color)?) };
        //let max_color = if max_color.is_null() { None } else { Some(get_color(max_color)?) };
        //let invalid_color = if invalid_color.is_null() { None } else { Some(get_color(invalid_color)?) };

        Ok(Clamped(self.inner.apply_cmap(
            cmap,
            min_color.as_ref().map(|a| a.view()),
            max_color.as_ref().map(|a| a.view()),
            invalid_color.view(),
        ).downcast::<u8>().unwrap().into_raw_vec()))
    }

    /// Return the array converted to datatype `dtype`. Throws an error if the conversion is not possible.
    pub fn astype(&self, dtype: &DataTypeLike) -> Result<JsArray, String> {
        Ok(self.inner.cast(dtype.try_into()?).into_owned().into())
    }

    /// Reshape array into shape `shape`.
    /// 
    /// Up to one axis can be specified as '-1', allowing it to be inferred from the length of the array.
    pub fn reshape(&self, shape: AxesLike) -> Result<JsArray, String> {
        let shape: Box<[isize]> = serde_wasm_bindgen::from_value(shape.obj).map_err(|e| e.to_string())?;

        Ok(self.inner.reshape(&shape)?.into())
    }

    #[wasm_bindgen]
    /// Return a contiguous, flattened array.
    pub fn ravel(&self) -> Result<JsArray, String> {
        Ok(self.inner.ravel().into())
    }

    #[wasm_bindgen]
    /// Return a contiguous, flattened array.
    pub fn flatten(&self) -> Result<JsArray, String> {
        Ok(self.inner.ravel().into())
    }

    #[wasm_bindgen(js_name = toNestedArray)]
    pub fn to_nested_array(&self) -> Result<NestedArray, String> {
        to_nested_array(&self.inner).map(|v| v.into())
    }

    #[wasm_bindgen(variadic, skip_typescript)]
    pub fn slice(&self, idxs: &JsValue) -> Result<JsArray, String> {
        let idxs: Vec<SliceInfoElem> = idxs.dyn_ref::<js_sys::Array>().ok_or_else(|| "'idxs' must be an array of indices or slices".to_owned())?
            .iter().map(|v| parse_index(&v)).try_collect()?;

        self.inner.slice(&idxs).map(|v| v.into())
    }
}

impl AsRef<DynArray> for JsArray {
    fn as_ref(&self) -> &DynArray { &self.inner }
}

impl std::ops::Deref for JsArray {
    type Target = DynArray;

    fn deref(&self) -> &DynArray { &self.inner }
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
    let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
        None
    } else { Some(dtype.try_into()?) };

    parse_arraylike(arr, dtype).map(|val| val.into_owned().into())
}

#[wasm_bindgen]
/// Return an array filled with ones, of shape `shape` and dtype `dtype`
pub fn ones(shape: ShapeLike, dtype: &DataTypeLike) -> Result<JsArray, String> {
    let shape: Box<[usize]> = shape.try_into()?;
    let dtype = dtype.try_into()?;
    Ok(DynArray::ones(shape.as_ref(), dtype).into())
}

#[wasm_bindgen]
/// Return an array filled with zeros, of shape `shape` and dtype `dtype`
pub fn zeros(shape: ShapeLike, dtype: &DataTypeLike) -> Result<JsArray, String> {
    let shape: Box<[usize]> = shape.try_into()?;
    let dtype = dtype.try_into()?;
    Ok(DynArray::zeros(shape.as_ref(), dtype).into())
}

#[wasm_bindgen(skip_typescript)]
pub fn arange(start: f64, end: Option<f64>, dtype: &DataTypeLike) -> Result<JsArray, String> {
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
}

#[wasm_bindgen(skip_typescript)]
pub fn indices(shape: ShapeLike, dtype: &DataTypeLike, sparse: Option<bool>) -> Result<Vec<JsArray>, String> {
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
}

#[wasm_bindgen(skip_typescript)]
pub fn linspace(start: f64, end: f64, n: usize, dtype: &DataTypeLike) -> Result<JsArray, String> {
    let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
        DataType::Float64
    } else { dtype.try_into()? };

    Ok(match dtype {
        DataType::Float32 => DynArray::linspace(start as f32, end as f32, n),
        DataType::Float64 => DynArray::linspace(start as f64, end as f64, n),
        dtype => return Err(format!("'linspace' not supported for dtype '{}'", dtype)),
    }.into())
}

#[wasm_bindgen(skip_typescript)]
pub fn logspace(start: f64, end: f64, n: usize, dtype: &DataTypeLike, base: Option<f64>) -> Result<JsArray, String> {
    let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
        DataType::Float64
    } else { dtype.try_into()? };
    let base = base.unwrap_or(10.);

    Ok(match dtype {
        DataType::Float32 => DynArray::logspace(start as f32, end as f32, n, base as f32),
        DataType::Float64 => DynArray::logspace(start as f64, end as f64, n, base),
        dtype => return Err(format!("'logspace' not supported for dtype '{}'", dtype)),
    }.into())
}

#[wasm_bindgen(skip_typescript)]
pub fn geomspace(start: f64, end: f64, n: usize, dtype: &DataTypeLike) -> Result<JsArray, String> {
    let dtype = if dtype.obj.is_undefined() || dtype.obj.is_null() {
        DataType::Float64
    } else { dtype.try_into()? };

    Ok(match dtype {
        DataType::Float32 => DynArray::geomspace(start as f32, end as f32, n),
        DataType::Float64 => DynArray::geomspace(start as f64, end as f64, n),
        dtype => return Err(format!("'geomspace' not supported for dtype '{}'", dtype)),
    }.into())
}

#[wasm_bindgen(skip_typescript)]
pub fn eye(ndim: f64, dtype: &DataTypeLike) -> Result<JsArray, String> {
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
}

// ## reshaping functions

#[wasm_bindgen]
/// Roll elements of the array by `shifts`, along axes `axes`.
///
/// If `axes` is specified, `shifts` and `axes` must be the same length.
/// Otherwise, `shifts` must be the same length as the array's dimensionality.
pub fn roll(arr: &ArrayLike, shifts: AxesLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    // TODO support single value shift arguments
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
}

#[wasm_bindgen]
/// Reshape array into shape `shape`.
/// 
/// Up to one axis can be specified as '-1', allowing it to be inferred from the length of the array.
pub fn reshape(arr: &ArrayLike, shape: AxesLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let shape: Box<[isize]> = serde_wasm_bindgen::from_value(shape.obj).map_err(|e| e.to_string())?;

    Ok(arr.as_ref().reshape(&shape)?.into())
}

#[wasm_bindgen]
/// Return a contiguous, flattened array.
pub fn ravel(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().ravel().into())
}

#[wasm_bindgen(variadic, skip_typescript)]
pub fn meshgrid(arrs: &JsValue) -> Result<Vec<JsArray>, String> {
    let arrs_arr: Vec<_> = arrs.dyn_ref::<js_sys::Array>().ok_or_else(|| "'arrs' must be an array of arrays".to_owned())?
        .iter().collect();
    let arrs: Vec<_> = arrs_arr.iter().map(|val| parse_arraylike(&val, None).map(|v| v.into_owned())).try_collect()?;

    DynArray::meshgrid(arrs, false).map(|v| v.into_iter().map(|arr| arr.into()).collect())
}

#[wasm_bindgen]
/// Stack arrays together along a new axis
pub fn stack(arrs: &JsValue, axis: Option<isize>) -> Result<JsArray, String> {
    let arrs_arr: Vec<_> = arrs.dyn_ref::<js_sys::Array>().ok_or_else(|| "'arrs' must be an array of arrays".to_owned())?
        .iter().collect();
    let arrs: Vec<_> = arrs_arr.iter().map(|arr| parse_arraylike(&arr, None)).try_collect()?;

    array::stack(arrs.iter().map(|arr| arr.as_ref()), axis.unwrap_or(0)).map(|arr| arr.into())
}

#[wasm_bindgen]
/// Split an array along a given axis. Defaults to the first axis
pub fn split(arr: &ArrayLike, axis: Option<isize>) -> Result<Vec<JsArray>, String> {
    let arr = parse_arraylike(arr, None)?;
    arr.split(axis.unwrap_or(0)).map(|arrs| arrs.into_iter().map(|arr| arr.into()).collect())
}

#[wasm_bindgen]
/// Broadcast an array to the given shape
pub fn broadcast_to(array: &ArrayLike, shape: ShapeLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(array, None)?.into_owned();
    let shape: Vec<usize> = TryInto::<Box<[usize]>>::try_into(shape)?.into();

    arr.broadcast_to(shape).map(|arr| arr.into()).map_err(|err| err.to_string())
}


#[wasm_bindgen(variadic, skip_typescript)]
/// Broadcast arrays together
pub fn broadcast_arrays(arrs: &JsValue) -> Result<Vec<JsArray>, String> {
    let arrs_arr: Vec<_> = arrs.dyn_ref::<js_sys::Array>().ok_or_else(|| "'arrs' must be an array of arrays".to_owned())?
        .iter().collect();
    let arrs: Vec<_> = arrs_arr.iter().map(|val| parse_arraylike(&val, None)).try_collect()?;
    let arr_refs: Vec<_> = arrs.iter().map(|arr| arr.as_ref()).collect();

    array::broadcast_arrays(&arr_refs).map(|arrs| arrs.into_iter().map(|arr| arr.into()).collect())
}

#[wasm_bindgen(variadic, skip_typescript)]
/// Broadcast shapes together
pub fn broadcast_shapes(arrs: &JsValue) -> Result<js_sys::Array, String> {
    let shapes: Vec<_> = arrs.dyn_ref::<js_sys::Array>().ok_or_else(|| "'shapes' must be an array of shapes".to_owned())?
        .iter().collect();
    let shapes: Vec<Box<[usize]>> = shapes.into_iter().map(|sh| ShapeLike::unchecked_from_js(sh).try_into()).try_collect()?;
    let shape_refs: Vec<&[usize]> = shapes.iter().map(|sh| &sh[..]).collect();

    array::broadcast_shapes(&shape_refs).map(|sh| sh.into_iter().map(|x| JsValue::from_f64(x as f64)).collect())
}

// ## elementwise functions

#[wasm_bindgen]
/// Return the ceiling of the input, element-wise
pub fn ceil(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().ceil().into())
}

#[wasm_bindgen]
/// Return the floor of the input, element-wise
pub fn floor(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().ceil().into())
}

#[wasm_bindgen]
/// Return the absolute value of the input, element-wise
pub fn abs(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().abs().into())
}

#[wasm_bindgen]
/// Return the complex conjugate of the input, element-wise
pub fn conj(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?; 
    Ok(arr.as_ref().conj().into())
}

#[wasm_bindgen]
/// Return the complex argument of the input, element-wise
pub fn angle(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().angle().into())
}

#[wasm_bindgen]
/// Return the absolute value of the input squared, element-wise
pub fn abs2(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().abs2().into())
}

#[wasm_bindgen]
/// Return the exponential e^x of the input, element-wise
pub fn exp_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().exp().into())
}

#[wasm_bindgen]
/// Return the natural (base-e) logarithm of the input, element-wise
pub fn log_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().log().into())
}

#[wasm_bindgen]
/// Return the base-2 logarithm of the input, element-wise
pub fn log2_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().log2().into())
}

#[wasm_bindgen]
/// Return the base-10 logarithm of the input, element-wise
pub fn log10_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().log10().into())
}

#[wasm_bindgen]
/// Return the square root of the input, element-wise
pub fn sqrt(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().sqrt().into())
}

#[wasm_bindgen]
/// Return the smallest element between the two arrays, elementwise.
/// 
/// Propagates NaNs, preferring the first value if both are NaN.
pub fn minimum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    let arr1 = parse_arraylike(arr1, None)?;
    let arr2 = parse_arraylike(arr2, None)?;
    Ok(arr1.as_ref().minimum(arr2.as_ref()).into())
}

#[wasm_bindgen]
/// Return the largest element between the two arrays, elementwise.
/// 
/// Propagates NaNs, preferring the first value if both are NaN.
pub fn maximum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    let arr1 = parse_arraylike(arr1, None)?;
    let arr2 = parse_arraylike(arr2, None)?;
    Ok(arr1.as_ref().maximum(arr2.as_ref()).into())
}

#[wasm_bindgen]
/// Return the smallest element between the two arrays, elementwise.
/// 
/// Ignores NaNs. If both values are NaN, returns the first one.
pub fn nanminimum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    let arr1 = parse_arraylike(arr1, None)?;
    let arr2 = parse_arraylike(arr2, None)?;
    Ok(arr1.as_ref().nanminimum(arr2.as_ref()).into())
}

#[wasm_bindgen]
/// Return the largest element between the two arrays, elementwise.
///
/// Ignores NaNs. If both values are NaN, returns the first one.
pub fn nanmaximum(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<JsArray, String> {
    let arr1 = parse_arraylike(arr1, None)?;
    let arr2 = parse_arraylike(arr2, None)?;
    Ok(arr1.as_ref().nanmaximum(arr2.as_ref()).into())
}

#[wasm_bindgen]
/// Return the sine of the input (in radians), element-wise
pub fn sin_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().sin().into())
}

#[wasm_bindgen]
/// Return the cosine of the input (in radians), element-wise
pub fn cos_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().cos().into())
}

#[wasm_bindgen]
/// Return the tangent of the input (in radians), element-wise
pub fn tan_(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().tan().into())
}

#[wasm_bindgen]
/// Return the arcsine of the input, element-wise. Returns a value in radians
pub fn arcsin(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().arcsin().into())
}

#[wasm_bindgen]
/// Return the arccosine of the input, element-wise. Returns a value in radians
pub fn arccos(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().arccos().into())
}

#[wasm_bindgen]
/// Return the arctangent of the input, element-wise. Returns a value in radians
pub fn arctan(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(arr.as_ref().arctan().into())
}

#[wasm_bindgen]
/// Return the arctangent of the inputs `y` and `x`, element-wise. Returns a value in radians
pub fn arctan2(y: &ArrayLike, x: &ArrayLike) -> Result<JsArray, String> {
    let y = parse_arraylike(y, None)?;
    let x = parse_arraylike(x, None)?;
    Ok(y.as_ref().arctan2(x.as_ref()).into())
}

// ## reductions

#[wasm_bindgen]
// Return the minimum element along the given axes.
//
// NaN values are propagated. See `nanmin` for a version that ignores missing values.
pub fn min(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    reductions::min(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
}

#[wasm_bindgen]
// Return the maximum element along the given axes.
//
// NaN values are propagated. See `nanmax` for a version that ignores missing values.
pub fn max(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    reductions::max(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
}

#[wasm_bindgen]
// Return the sum of elements along the given axes.
//
// NaN values are propagated. See `nansum` for a version that ignores missing values.
pub fn sum(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    Ok(reductions::sum(arr.as_ref(), axes.as_deref()).into())
}

#[wasm_bindgen]
// Return the product of elements along the given axes.
//
// NaN values are propagated. See `nanprod` for a version that ignores missing values.
pub fn prod(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    Ok(reductions::prod(arr.as_ref(), axes.as_deref()).into())
}

#[wasm_bindgen]
// Return the mean element along the given axes.
//
// NaN values are propagated. See `nanmean` for a version that ignores missing values.
pub fn mean(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    Ok(reductions::mean(arr.as_ref(), axes.as_deref()).into())
}

#[wasm_bindgen]
// Return the minimum element along the given axes.
//
// NaN values are ignored.
pub fn nanmin(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    reductions::nanmin(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
}

#[wasm_bindgen]
// Return the maximum element along the given axes.
//
// NaN values are ignored.
pub fn nanmax(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    reductions::nanmax(arr.as_ref(), axes.as_deref()).map(|arr| arr.into())
}

#[wasm_bindgen]
// Return the sum of elements along the given axes.
//
// NaN values are ignored.
pub fn nansum(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    Ok(reductions::nansum(arr.as_ref(), axes.as_deref()).into())
}

#[wasm_bindgen]
// Return the product of elements along the given axes.
//
// NaN values are ignored.
pub fn nanprod(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    Ok(reductions::nanprod(arr.as_ref(), axes.as_deref()).into())
}

#[wasm_bindgen]
// Return the mean element along the given axes.
//
// NaN values are ignored.
pub fn nanmean(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes = axes.map(|val| {
        serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
    }).transpose()?;

    Ok(reductions::nanmean(arr.as_ref(), axes.as_deref()).into())
}

// ## FFT functions

#[wasm_bindgen]
/// Compute the Fourier transform of the input array
/// 
/// Computes the transformation along each of `axes` (defaults to all axes).
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn fft(arr: &ArrayLike, axes: Option<AxesLike>, norm: Option<FFTNorm>) -> Result<JsArray, String> {
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
}

#[wasm_bindgen]
/// Compute the inverse Fourier transform of the input array
/// 
/// Computes the transformation along each of `axes` (defaults to all axes).
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn ifft(arr: &ArrayLike, axes: Option<AxesLike>, norm: Option<FFTNorm>) -> Result<JsArray, String> {
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
}

#[wasm_bindgen]
/// Shifts the zero-frequency component of a Fourier transformed array to the center
/// 
/// Shifts along each of `axes` (defaults to all axes).
pub fn fftshift(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes: Box<[isize]> = match axes {
        None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
        Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
    };

    Ok(fft::fftshift(arr.as_ref(), &axes).into())
}

#[wasm_bindgen]
/// Inverse of `fftshift`. Shifts the zero-frequency component of a Fourier transformed array to the corner
/// 
/// Shifts along each of `axes` (defaults to all axes).
pub fn ifftshift(arr: &ArrayLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let axes: Box<[isize]> = match axes {
        None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
        Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
    };

    Ok(fft::ifftshift(arr.as_ref(), &axes).into())
}

#[wasm_bindgen]
/// Compute the Fourier transform of the input array
/// 
/// Computes the transformation along the last two axes of the input.
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn fft2(arr: &ArrayLike, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let norm = norm.map(|norm| match norm.obj.as_string() {
        Some(s) => fft::FFTNorm::try_from(s.as_ref()),
        None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
    }).transpose()?;

    Ok(fft::fft(arr.as_ref(), &[-2, -1], norm).into())
}

#[wasm_bindgen]
/// Compute the inverse Fourier transform of the input array
/// 
/// Computes the transformation along the last two axes of the input.
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn ifft2(arr: &ArrayLike, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    let norm = norm.map(|norm| match norm.obj.as_string() {
        Some(s) => fft::FFTNorm::try_from(s.as_ref()),
        None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
    }).transpose()?;

    Ok(fft::ifft(arr.as_ref(), &[-2, -1], norm).into())
}

#[wasm_bindgen]
/// Shifts the zero-frequency component of a Fourier transformed array to the center
/// 
/// Computes the transformation along the last two axes of the input.
pub fn fft2shift(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(fft::fftshift(arr.as_ref(), &[-2, -1]).into())
}

#[wasm_bindgen]
/// Inverse of `fft2shift`. Shifts the zero-frequency component of a Fourier transformed array to the corner
/// 
/// Computes the transformation along the last two axes of the input.
pub fn ifft2shift(arr: &ArrayLike) -> Result<JsArray, String> {
    let arr = parse_arraylike(arr, None)?;
    Ok(fft::ifftshift(arr.as_ref(), &[-2, -1]).into())
}

// ## reductions to bool

#[wasm_bindgen]
pub fn allequal(arr1: &ArrayLike, arr2: &ArrayLike) -> Result<bool, String> {
    let arr1 = parse_arraylike(arr1, None)?;
    let arr2 = parse_arraylike(arr2, None)?;
    Ok(arr1.as_ref().allequal(arr2.as_ref()))
}

#[wasm_bindgen]
pub fn allclose(arr1: &ArrayLike, arr2: &ArrayLike, rtol: Option<f64>, atol: Option<f64>) -> Result<bool, String> {
    let arr1 = parse_arraylike(arr1, None)?;
    let arr2 = parse_arraylike(arr2, None)?;
    Ok(arr1.as_ref().allclose(arr2.as_ref(), rtol.unwrap_or(1e-8), atol.unwrap_or(0.0)))
}

// ## special functions

#[wasm_bindgen(skip_typescript)]
/// 1D linear interpolation.
/// Interpolates a set of values `xs` onto the piecewise line defined by `xp` and `yp`.
/// 
/// `xp` and `yp` must be 1D arrays. `xp` must be sorted and must not contain NaN values.
/// `left` and `right` define values to return in the case of `xs` outside `xp`. They default to the
/// first and last element of `yp` respectively.
/// 
/// Returns an array of the same shape as `xs`.
pub fn interp(xs: &ArrayLike, xp: &ArrayLike, yp: &ArrayLike, left: Option<f64>, right: Option<f64>) -> Result<JsArray, String> {
    let xs = parse_arraylike(xs, None)?;
    let xp = parse_arraylike(xp, None)?;
    let yp = parse_arraylike(yp, None)?;
    array::interp(xs.as_ref(), xp.as_ref(), yp.as_ref(), left, right).map(|arr| arr.into())
}

#[wasm_bindgen]
pub fn interpn(coords: &JsValue, values: &ArrayLike, xs: &ArrayLike, fill: Option<f64>) -> Result<JsArray, String> {
    let coords_arr: Vec<_> = coords.dyn_ref::<js_sys::Array>().ok_or_else(|| "'coords' must be an array of 1D coordinate arrays".to_owned())?
        .iter().collect();
    let coords: Vec<_> = coords_arr.iter().map(|arr| parse_arraylike(&arr, None)).try_collect()?;

    let values = parse_arraylike(values, None)?;
    let xs = parse_arraylike(xs, None)?;

    array::interpn(&coords.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>(), &values, &xs, fill).map(|arr| arr.into())
}

// ## from_interchange

#[wasm_bindgen]
/// Create an array from a JSON interchange format, loosely conforming with numpy's __array_interface__ protocol.
pub fn from_interchange(obj: IArrayInterchange) -> Result<JsArray, String> {
    Ok(ArrayInterchange::try_from(obj)?.to_array()?.into())
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
        Box::new(UnaryFunc::new("abs", DynArray::abs)),
        Box::new(UnaryFunc::new("abs2", DynArray::abs2)),
        Box::new(UnaryFunc::new("exp", DynArray::exp)),
        Box::new(UnaryFunc::new("log", DynArray::log)),
        Box::new(UnaryFunc::new("log2", DynArray::log2)),
        Box::new(UnaryFunc::new("log10", DynArray::log10)),
        Box::new(UnaryFunc::new("sqrt", DynArray::sqrt)),
        Box::new(UnaryFunc::new("ceil", DynArray::ceil)),
        Box::new(UnaryFunc::new("floor", DynArray::floor)),
        Box::new(BinaryFunc::new("minimum", |l, r| l.minimum(r))),
        Box::new(BinaryFunc::new("maximum", |l, r| l.maximum(r))),
        Box::new(BinaryFunc::new("nanminimum", |l, r| l.nanminimum(r))),
        Box::new(BinaryFunc::new("nanmaximum", |l, r| l.nanmaximum(r))),

        Box::new(UnaryFunc::new("sin", DynArray::sin)),
        Box::new(UnaryFunc::new("cos", DynArray::cos)),
        Box::new(UnaryFunc::new("tan", DynArray::tan)),
        Box::new(UnaryFunc::new("arcsin", DynArray::arcsin)),
        Box::new(UnaryFunc::new("arccos", DynArray::arccos)),
        Box::new(UnaryFunc::new("arctan", DynArray::arctan)),
        Box::new(BinaryFunc::new("arctan2", |y, x| y.arctan2(x))),
        Box::new(UnaryFunc::new("conj", DynArray::conj)),
        Box::new(UnaryFunc::new("angle", DynArray::angle)),
    ];

    funcs.into_iter().map(|f| (f.name(), f)).collect()
}

#[wasm_bindgen(variadic, skip_typescript)]
pub fn expr(strs: Vec<String>, lits: &JsValue) -> Result<JsArray, String> {
        set_panic_hook();
        let lits = lits.clone().dyn_into::<js_sys::Array>().map_err(|_| "'lits' must be an array".to_owned())?;
        let lits: Vec<_> = lits.iter().map(|val| parse_arraylike(&val, None).map(|v| v.into_owned())).try_collect()?;

        let funcs = ARRAY_FUNCS.get_or_init(init_array_funcs);
        //return Err(format!("strs: {:?} lits: {:?}", strs, lits.into_iter().map(|a| a.inner).collect_vec()));
        let expr = parse_with_literals(strs.iter().map(|s| s.as_ref()), lits.into_iter().map(Token::ArrayLit))
            .map_err(|e| format!("Parse error: {:?}", e))?;

        let vars = HashMap::new();
        match expr.exec(&vars, funcs) {
            Ok(arr) => Ok(arr.into()),
            Err(e) => Err(format!("{:?}", e)),
        }
}

pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
    //log::subscribe(Box::new(_log));
}

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    set_panic_hook();
    Ok(())
}
