#![feature(iterator_try_collect)]
#![feature(try_blocks)]

extern crate alloc;

use core::fmt;
use std::collections::HashMap;
use std::panic::{catch_unwind, UnwindSafe};
use std::sync::OnceLock;

use arraylib::bool::Bool;
use num::Complex;
use serde::{Deserialize, Serialize, de, ser};
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use ndarray::Array1;

pub mod expr;

use expr::{parse_with_literals, Token, ArrayFunc, UnaryFunc, FuncMap};

use arraylib::array::DynArray;
use arraylib::dtype::DataType;
use arraylib::error::ArrayError;
use arraylib::colors::named_colors;
use arraylib::{fft, reductions};

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
 * Array-like. Can be a number, `BigInt`, typed array, or nested array.
 */
type ArrayLike = NArray | NestedArray<number> | NestedArray<BigInt> | ArrayBufferView;

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
export function expr(strs: ReadonlyArray<string>, ...lits: ReadonlyArray<NArray>): NArray;

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

pub trait DowncastWasmExport: wasm_bindgen::convert::RefFromWasmAbi<Abi = u32> {

    fn __get_type_id() -> &'static str;

    fn downcast_ref(js: &JsValue) -> Option<<Self as wasm_bindgen::convert::RefFromWasmAbi>::Anchor> {
        if !js.is_object() { return None; }

        let js_typeid_function = match js_sys::Reflect::get(js, &JsValue::from("__getTypeId")).map(|val| val.dyn_into::<js_sys::Function>()) {
            Ok(Ok(val)) => val,
            _ => return None,
        };
        let js_type_id = match js_sys::Reflect::apply(&js_typeid_function, js, &::js_sys::Array::new()).map(|val| val.as_string()) {
            Ok(Some(val)) => val,
            _ => return None,
        };
        if js_type_id != JsDataType::__get_type_id() { return None; }

        let ptr = js_sys::Reflect::get(js, &JsValue::from_str("__wbg_ptr")).unwrap().as_f64().expect("Null object reference") as u32;
        unsafe { Some(Self::ref_from_abi(ptr)) }
    }

    fn downcast_option_ref(js: &JsValue) -> Option<Option<<Self as wasm_bindgen::convert::RefFromWasmAbi>::Anchor>> {
        if js.is_undefined() || js.is_null() {
            None
        } else {
            Some(Self::downcast_ref(js))
        }
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(typescript_type = "IArrayInterchange")]
    pub type IArrayInterchange;

    #[wasm_bindgen(typescript_type = "ColorLike")]
    pub type ColorLike;

    #[wasm_bindgen(typescript_type = "DataTypeLike")]
    pub type DataTypeLike;

    #[wasm_bindgen(typescript_type = "ShapeLike")]
    pub type ShapeLike;

    #[wasm_bindgen(typescript_type = "AxesLike")]
    pub type AxesLike;

    #[wasm_bindgen(typescript_type = "FFTNorm")]
    pub type FFTNorm;
}

impl TryInto<Array1<f32>> for ColorLike {
    type Error = String;

    fn try_into(self) -> Result<Array1<f32>, Self::Error> {
        if let Some(s) = self.obj.as_string() {
            if s.as_bytes().first() == Some(&b'#') {
                if s.len() == 7 || s.len() == 11 {
                    if let Ok(v) = u32::from_str_radix(&s[1..], 16) {
                        if s.len() == 7 { // #rrggbb
                            return Ok(Array1::from_vec(vec![
                                ((v & 0xff0000) >> 16) as f32,
                                ((v & 0x00ff00) >> 8) as f32,
                                ((v & 0x0000ff) >> 0) as f32,
                            ]) / 255.0);
                        }
                        // #rrggbbaa
                        return Ok(Array1::from_vec(vec![
                            ((v & 0xff000000) >> 24) as f32,
                            ((v & 0x00ff0000) >> 16) as f32,
                            ((v & 0x0000ff00) >> 8) as f32,
                            ((v & 0x000000ff) >> 0) as f32,
                        ]) / 255.0);
                    }
                }
                return Err(format!("Invalid color string '{}'", s));
            } else {
                return named_colors().get(s.as_str()).ok_or_else(|| format!("Unknown color '{}'", s))
                    .map(|arr_view| arr_view.to_owned());
            };
        }
        if let Ok(arr) = self.obj.dyn_into::<js_sys::Array>() {
            let len = arr.length();
            if len >= 3 && len <= 4 {
                return Err("Invalid color array. Expected an array of 3 or 4 floats between 0 and 1".to_owned());
            }
            let vals: Vec<f32> = arr.into_iter().map(|v| {
                if let Some(v) = v.as_f64() {
                    if v > 0. && v < 1. {
                        return Ok(v as f32);
                    }
                }
                Err("Invalid color array. Expected an array of floats between 0 and 1".to_owned())
            }).try_collect()?;
            return Ok(Array1::from_vec(vals));
        }
        Err("Invalid color. Expected an array of floats or a string".to_owned())
    }
}

impl<'a> TryInto<DataType> for &'a DataTypeLike {
    type Error = String;

    fn try_into(self) -> Result<DataType, Self::Error> {
        if let Some(val) = JsDataType::downcast_ref(&self.obj) {
            return Ok(val.inner);
        }
        let dtype = self.obj.as_string().ok_or_else(|| format!("Invalid datatype. Expected a DataType object or string, instead got object {:?} of type {:?}", self.obj, self.obj.js_typeof().as_string().unwrap()))?;
        //let dtype = obj.as_string().ok_or_else(|| format!("Invalid datatype. Expected a DataType object or string"))?;
        Ok(match dtype.to_lowercase().as_ref() {
            "bool" => DataType::Boolean,
            "uint8" | "u8" => DataType::UInt8,
            "uint16" | "u16" => DataType::UInt16,
            "uint32" | "u32" => DataType::UInt32,
            "uint64" | "u64" => DataType::UInt64,
            "int8" | "i8" => DataType::Int8,
            "int16" | "i16" => DataType::Int16,
            "int32" | "i32" => DataType::Int32,
            "int64" | "i64" | "int" => DataType::Int64,
            "float32" | "f32" => DataType::Float32,
            "float64" | "f64" | "float" => DataType::Float64,
            "complex64" | "c64" => DataType::Complex64,
            "complex128" | "c128" | "complex" => DataType::Complex128,
            _ => return Err(format!("Unknown datatype '{}'", dtype)),
        })
    }
}

impl TryInto<DataType> for DataTypeLike {
    type Error = String;

    fn try_into(self) -> Result<DataType, Self::Error> { (&self).try_into() }
}

impl TryInto<Box<[usize]>> for ShapeLike {
    type Error = String;

    fn try_into(self) -> Result<Box<[usize]>, Self::Error> {
        serde_wasm_bindgen::from_value::<Box<[usize]>>(self.obj).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArrayInterchange {
    #[serde(rename = "data", with = "serde_bytes")]
    pub buf: Box<[u8]>,
    #[serde(rename = "typestr", serialize_with = "serialize_typestr", deserialize_with = "deserialize_typestr")]
    pub dtype: DataType,
    pub shape: Box<[usize]>,
    pub strides: Option<Box<[isize]>>,
    #[serde(default = "default_version")]
    pub version: u32,
}

impl TryFrom<IArrayInterchange> for ArrayInterchange {
    type Error = String;

    fn try_from(value: IArrayInterchange) -> Result<Self, Self::Error> {
        serde_wasm_bindgen::from_value::<ArrayInterchange>(value.obj).map_err(|e| e.to_string())
    }
}

fn default_version() -> u32 { 3 }

fn serialize_typestr<S: ser::Serializer>(dtype: &DataType, ser: S) -> Result<S::Ok, S::Error> {
    ser.serialize_str(to_typestr(dtype))
}

struct TypestrVisitor;

impl<'de> de::Visitor<'de> for TypestrVisitor {
    type Value = DataType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Array typestring")
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<DataType, E> {
        parse_typestr(v).map_err(|e| de::Error::custom(e))
    }
}

fn deserialize_typestr<'de, D: de::Deserializer<'de>>(de: D) -> Result<DataType, D::Error> {
    de.deserialize_str(TypestrVisitor)
}

impl ArrayInterchange {
    pub fn to_array(self) -> Result<DynArray, String> {
        DynArray::from_buf(self.buf, self.dtype, self.shape, self.strides)
    }
}

impl From<DynArray> for ArrayInterchange {
    fn from(value: DynArray) -> Self {
        let (buf, dtype, shape, strides) = value.to_buf();
        Self {
            buf, dtype, shape, strides, version: 3,
        }
    }
}

fn parse_typestr(s: &str) -> Result<DataType, String> {
    match s.chars().next() {
        Some('<' | '|') => (),
        Some('>') => return Err(format!("Only little-ending types are supported, instead got typestring '{}'", s)),
        Some(byteorder) => return Err(format!("Invalid byteorder char '{byteorder}' in typestring '{}'", s)),
        None => return Err("Empty typestring".to_owned()),
    }
    Ok(match &s[1..] {
        "u1" => DataType::UInt8,
        "u2" => DataType::UInt16,
        "u4" => DataType::UInt32,
        "u8" => DataType::UInt64,
        "i1" => DataType::Int8,
        "i2" => DataType::Int16,
        "i4" => DataType::Int32,
        "i8" => DataType::Int64,
        "f4" => DataType::Float32,
        "f8" => DataType::Float64,
        "c8" => DataType::Complex64,
        "c16" => DataType::Complex128,
        "b1" => DataType::Boolean,
        _ => return Err(format!("Invalid/unsupported typestring '{}'", s)),
    })
}

fn to_typestr(datatype: &DataType) -> &'static str {
    match datatype {
        DataType::UInt8 => "|u1",
        DataType::UInt16 => "<u2",
        DataType::UInt32 => "<u4",
        DataType::UInt64 => "<u8",
        DataType::Int8 => "|i1",
        DataType::Int16 => "<i2",
        DataType::Int32 => "<i4",
        DataType::Int64 => "<i8",
        DataType::Float32 => "<f4",
        DataType::Float64 => "<f8",
        DataType::Complex64 => "<c8",
        DataType::Complex128 => "<c16",
        DataType::Boolean => "|b1",
    }
}

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

    #[wasm_bindgen(js_name = "__getTypeId")]
    pub fn __js_get_type_id(&self) -> String {
        "HTxHNbDq84KHGdf5".to_owned()
    }
}

impl DowncastWasmExport for JsDataType {
    fn __get_type_id() -> &'static str { "HTxHNbDq84KHGdf5" }
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
        reshape(self, shape)
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

    #[wasm_bindgen(js_name = "__getTypeId")]
    pub fn __js_get_type_id(&self) -> String {
        "dbvjUb9K4VrV2Fq5".to_owned()
    }
}

impl DowncastWasmExport for JsArray {
    fn __get_type_id() -> &'static str { "dbvjUb9K4VrV2Fq5" }
}

#[wasm_bindgen]
/// Create a `DataType` object from a `DataTypeLike`.
pub fn to_dtype(dtype: &DataTypeLike) -> Result<JsDataType, String> {
    Ok(<&DataTypeLike as TryInto<DataType>>::try_into(dtype)?.into())
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

#[wasm_bindgen]
/// Return the ceiling of the input, element-wise
pub fn ceil(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| { Ok(arr.inner.ceil().into()) })
}

#[wasm_bindgen]
/// Return the floor of the input, element-wise
pub fn floor(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| { Ok(arr.inner.ceil().into()) })
}

#[wasm_bindgen]
/// Return the absolute value of the input, element-wise
pub fn abs(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| { Ok(arr.inner.abs().into()) })
}

#[wasm_bindgen]
/// Return the complex conjugate of the input, element-wise
pub fn conj(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| { Ok(arr.inner.conj().into()) })
}

#[wasm_bindgen]
/// Return the square root of the input, element-wise
pub fn sqrt(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| { Ok(arr.inner.sqrt().into()) })
}

#[wasm_bindgen]
/// Roll elements of the array by `shifts`, along axes `axes`.
///
/// If `axes` is specified, `shifts` and `axes` must be the same length.
/// Otherwise, `shifts` must be the same length as the array's dimensionality.
pub fn roll(arr: &JsArray, shifts: AxesLike, axes: Option<AxesLike>) -> Result<JsArray, String> {
    // TODO support single value shift arguments
    catch_panic(|| {
        let shifts: Box<[isize]> = serde_wasm_bindgen::from_value(shifts.obj).map_err(|e| e.to_string())?;

        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };

        if shifts.len() != axes.len() {
            return Err(format!("'shifts' must be the same length as 'axes' (or the array's ndim, if 'axes' is not specified)."))
        }

        Ok(arr.inner.roll(&shifts, &axes).into())
    })
}

#[wasm_bindgen]
/// Reshape array into shape `shape`.
/// 
/// Up to one axis can be specified as '-1', allowing it to be inferred from the length of the array.
pub fn reshape(arr: &JsArray, shape: AxesLike) -> Result<JsArray, String> {
    catch_panic(|| {
        let shape: Box<[isize]> = serde_wasm_bindgen::from_value(shape.obj).map_err(|e| e.to_string())?;

        Ok(arr.inner.reshape(&shape)?.into())
    })
}

#[wasm_bindgen]
/// Return a contiguous, flattened array.
pub fn ravel(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| {
        Ok(arr.inner.ravel().into())
    })
}

#[wasm_bindgen]
// Return the minimum element along the given axes.
//
// NaN values are propagated. See `nanmin` for a version that ignores missing values.
pub fn min(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::min(&arr.inner, axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the maximum element along the given axes.
//
// NaN values are propagated. See `nanmax` for a version that ignores missing values.
pub fn max(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::max(&arr.inner, axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the sum of elements along the given axes.
//
// NaN values are propagated. See `nansum` for a version that ignores missing values.
pub fn sum(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::sum(&arr.inner, axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the product of elements along the given axes.
//
// NaN values are propagated. See `nanprod` for a version that ignores missing values.
pub fn prod(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::prod(&arr.inner, axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the mean element along the given axes.
//
// NaN values are propagated. See `nanmean` for a version that ignores missing values.
pub fn mean(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::mean(&arr.inner, axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the minimum element along the given axes.
//
// NaN values are ignored.
pub fn nanmin(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::nanmin(&arr.inner, axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the maximum element along the given axes.
//
// NaN values are ignored.
pub fn nanmax(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        reductions::nanmax(&arr.inner, axes.as_deref()).map(|arr| arr.into())
    })
}

#[wasm_bindgen]
// Return the sum of elements along the given axes.
//
// NaN values are ignored.
pub fn nansum(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::nansum(&arr.inner, axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the product of elements along the given axes.
//
// NaN values are ignored.
pub fn nanprod(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::nanprod(&arr.inner, axes.as_deref()).into())
    })
}

#[wasm_bindgen]
// Return the mean element along the given axes.
//
// NaN values are ignored.
pub fn nanmean(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes = axes.map(|val| {
            serde_wasm_bindgen::from_value::<Box<[isize]>>(val.obj).map_err(|e| e.to_string())
        }).transpose()?;

        Ok(reductions::nanmean(&arr.inner, axes.as_deref()).into())
    })
}

#[wasm_bindgen]
/// Compute the Fourier transform of the input array
/// 
/// Computes the transformation along each of `axes` (defaults to all axes).
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn fft(arr: &JsArray, axes: Option<AxesLike>, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::fft(&arr.inner, &axes, norm).into())
    })
}

#[wasm_bindgen]
/// Compute the inverse Fourier transform of the input array
/// 
/// Computes the transformation along each of `axes` (defaults to all axes).
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn ifft(arr: &JsArray, axes: Option<AxesLike>, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::ifft(&arr.inner, &axes, norm).into())
    })
}

#[wasm_bindgen]
/// Shifts the zero-frequency component of a Fourier transformed array to the center
/// 
/// Shifts along each of `axes` (defaults to all axes).
pub fn fftshift(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };

        Ok(fft::fftshift(&arr.inner, &axes).into())
    })
}

#[wasm_bindgen]
/// Inverse of `fftshift`. Shifts the zero-frequency component of a Fourier transformed array to the corner
/// 
/// Shifts along each of `axes` (defaults to all axes).
pub fn ifftshift(arr: &JsArray, axes: Option<AxesLike>) -> Result<JsArray, String> {
    catch_panic(|| {
        let axes: Box<[isize]> = match axes {
            None => (0..arr.shape().len()).into_iter().map(|v| v as isize).collect(),
            Some(val) => serde_wasm_bindgen::from_value(val.obj).map_err(|e| e.to_string())?,
        };

        Ok(fft::ifftshift(&arr.inner, &axes).into())
    })
}

#[wasm_bindgen]
/// Compute the Fourier transform of the input array
/// 
/// Computes the transformation along the last two axes of the input.
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn fft2(arr: &JsArray, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::fft(&arr.inner, &[-2, -1], norm).into())
    })
}

#[wasm_bindgen]
/// Compute the inverse Fourier transform of the input array
/// 
/// Computes the transformation along the last two axes of the input.
/// Uses the normalization `norm`, which can be `'backward'` (default), `'forward'`, or `'ortho'`.
pub fn ifft2(arr: &JsArray, norm: Option<FFTNorm>) -> Result<JsArray, String> {
    catch_panic(|| {
        let norm = norm.map(|norm| match norm.obj.as_string() {
            Some(s) => fft::FFTNorm::try_from(s.as_ref()),
            None => Err(format!("Expected a string 'backward', 'forward', or 'ortho', got type {} instead", norm.obj.js_typeof().as_string().unwrap())),
        }).transpose()?;

        Ok(fft::ifft(&arr.inner, &[-2, -1], norm).into())
    })
}

#[wasm_bindgen]
/// Shifts the zero-frequency component of a Fourier transformed array to the center
/// 
/// Computes the transformation along the last two axes of the input.
pub fn fft2shift(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| {
        Ok(fft::fftshift(&arr.inner, &[-2, -1]).into())
    })
}

#[wasm_bindgen]
/// Inverse of `fft2shift`. Shifts the zero-frequency component of a Fourier transformed array to the corner
/// 
/// Computes the transformation along the last two axes of the input.
pub fn ifft2shift(arr: &JsArray) -> Result<JsArray, String> {
    catch_panic(|| {
        Ok(fft::ifftshift(&arr.inner, &[-2, -1]).into())
    })
}

#[wasm_bindgen]
/// Create an array from a JSON interchange format, loosely conforming with numpy's __array_interface__ protocol.
pub fn from_interchange(obj: IArrayInterchange) -> Result<JsArray, String> {
    catch_panic(|| {
        Ok(ArrayInterchange::try_from(obj)?.to_array()?.into())
    })
}

static ARRAY_FUNCS: OnceLock<FuncMap> = OnceLock::new();

fn init_array_funcs() -> FuncMap {
    let funcs: Vec<Box<dyn ArrayFunc + Sync + Send>> = vec![
        Box::new(UnaryFunc::new("abs", |v| v.abs())),
        Box::new(UnaryFunc::new("exp", |v| v.exp())),
        Box::new(UnaryFunc::new("sqrt", |v| v.sqrt())),
    ];

    funcs.into_iter().map(|f| (f.name(), f)).collect()
}

#[wasm_bindgen(variadic, skip_typescript)]
pub fn expr(strs: Vec<String>, lits: Vec<JsArray>) -> Result<JsArray, String> {
    catch_panic(|| {
        let funcs = ARRAY_FUNCS.get_or_init(init_array_funcs);
        //return Err(format!("strs: {:?} lits: {:?}", strs, lits.into_iter().map(|a| a.inner).collect_vec()));
        let expr = parse_with_literals(strs.iter().map(|s| s.as_ref()), lits.into_iter().map(|v| Token::ArrayLit(v.inner)))
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