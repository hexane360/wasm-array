#![feature(iterator_try_collect)]
#![feature(try_blocks)]

use core::fmt;
use std::collections::HashMap;
use std::panic::{catch_unwind, UnwindSafe};

use serde::{Deserialize, Serialize, de, ser};
use wasm_bindgen::convert::TryFromJsValue;
use wasm_bindgen::prelude::*;
use num::NumCast;

pub mod expr;

use expr::{parse_with_literals, Token};

use arraylib::array::DynArray;
use arraylib::dtype::DataType;
use arraylib::error::ArrayError;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen(js_name = DataType)]
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
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> JsDataType {
        self.inner.dtype().into()
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<js_sys::Object, JsValue> {
        let map = js_sys::Map::new();
        map.set(&"dtype".to_owned().into(), &self.inner.dtype().to_string().into());
        let shape = self.inner.shape();
        map.set(&"shape".to_owned().into(), &js_sys::Array::from_iter(shape.into_iter().map(|v| JsValue::from_f64(v as f64))));
        js_sys::Object::from_entries(&map.into())
    }

    #[wasm_bindgen(js_name = toInterchange)]
    pub fn to_interchange(self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&ArrayInterchange::from(self.inner)).map_err(|e| e.to_string())?)
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        let mut s = format!("Array {}\n{}", self.inner.dtype(), self.inner);
        if s.lines().count() < 3 {
            let i = s.find('\n').unwrap();
            // SAFETY: we replace an entire UTF-8 codepoint '\n' with ' '
            unsafe { s.as_mut_vec()[i] = b' ' }
        }
        s
    }
}

#[wasm_bindgen]
pub fn to_dtype(dtype: &str) -> Result<JsDataType, String> {
    Ok(match dtype.to_lowercase().as_str() {
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
    }.into())
}

pub struct ArrayBuffer {
    inner: Box<[u8]>,
}

impl Into<Box<[u8]>> for ArrayBuffer {
    fn into(self) -> Box<[u8]> { self.inner }
}

impl Into<Vec<u8>> for ArrayBuffer {
    fn into(self) -> Vec<u8> { self.inner.into_vec() }
}

impl TryFromJsValue for ArrayBuffer {
    type Error = String;

    fn try_from_js_value(value: JsValue) -> Result<Self, Self::Error> {
        if let Some(v) = value.dyn_ref::<js_sys::Uint8Array>() {
            return Ok(ArrayBuffer { inner: v.to_vec().into_boxed_slice() })
        }
        if let Some(v) = value.dyn_ref::<js_sys::ArrayBuffer>() {
            let v = js_sys::Uint8Array::new(v);
            return Ok(ArrayBuffer { inner: v.to_vec().into_boxed_slice() })
        }
        from_array_buf_view(value).ok_or_else(|| "Expected an ArrayBuffer or ArrayBufferView".to_owned())
    }
}

fn from_array_buf_view(value: JsValue) -> Option<ArrayBuffer> {
    let buffer = js_sys::Reflect::get(&value, &JsValue::from_str("buffer")).ok()?.dyn_into::<js_sys::ArrayBuffer>().ok()?;
    let byte_offset = js_sys::Reflect::get(&value, &JsValue::from_str("byte_offset")).ok()?.as_f64()? as u32;
    let byte_length = js_sys::Reflect::get(&value, &JsValue::from_str("byte_length")).ok()?.as_f64()? as u32;

    Some(ArrayBuffer {
        inner: js_sys::Uint8Array::new_with_byte_offset_and_length(&buffer, byte_offset, byte_length).to_vec().into_boxed_slice()
    })
}

fn get_object_field(obj: &JsValue, field: &'static str) -> Result<JsValue, String> {
    let v: JsValue = js_sys::Reflect::get(obj, &JsValue::from_str(field)).unwrap();
    if v.is_undefined() {
        Err(format!("Object missing required field '{}'", field))
    } else {
        Ok(v)
    }
}

macro_rules! to_const { ( $n:expr, $const:ty ) => { $const }; }
macro_rules! get_object_fields {
    ( $obj:ident, ($( $n:expr ),*) ) => { {
        let ret: Result<($( to_const!($n, JsValue) ),*), String> = try { ($( get_object_field(&$obj, $n)? ),*) };
        ret
    } };
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

impl TryFromJsValue for ArrayInterchange {
    type Error = String;

    fn try_from_js_value(value: JsValue) -> Result<Self, Self::Error> {
        if !value.is_object() { return Err("Expected an object".to_owned()) }
        let (buf, dtype, shape, strides) = get_object_fields!(value, ("data", "typestr", "shape", "strides"))?;

        let buf = ArrayBuffer::try_from_js_value(buf)?.into();
        let dtype = parse_typestr(&dtype.as_string().ok_or_else(|| "Field 'typestr' must be a string, and valid UTF-8".to_owned())?)?;

        let shape = shape.dyn_into::<js_sys::Array>().map_err(|_| "Field 'shape' must be an array".to_owned())?.into_iter()
            .map(|v| v.as_f64().and_then(|v| <usize as NumCast>::from(v))).try_collect().ok_or_else(|| "Field 'shape' must be an array of non-negative ints")?;

        let strides = if strides.is_null() { None } else {
            Some(strides.dyn_into::<js_sys::Array>().map_err(|_| "Field 'strides' must be an array".to_owned())?.into_iter()
                .map(|v| v.as_f64().and_then(|v| <isize as NumCast>::from(v))).try_collect().ok_or_else(|| "Field 'strides' must be an array of ints")?)
        };

        let version = match get_object_field(&value, "version") {
            Ok(v) => v.as_f64().ok_or_else(|| "Field 'version' must be a number".to_owned())? as u32,
            Err(_) => 3,
        };

        Ok(ArrayInterchange {
            buf, dtype, shape, strides, version
        })
    }
}

impl ArrayInterchange {
    pub fn into_js_object(self) -> js_sys::Object {
        let map = js_sys::Map::new();
        map.set(&"data".to_owned().into(), &js_sys::Uint8Array::from(self.buf.as_ref()).buffer());
        map.set(&"typestr".to_owned().into(), &to_typestr(&self.dtype).to_string().into());
        map.set(&"shape".to_owned().into(), &js_sys::Array::from_iter(self.shape.into_iter().map(|v| JsValue::from_f64(*v as f64))));
        map.set(&"version".to_owned().into(), &JsValue::from_f64(self.version as f64));

        let strides_obj = match self.strides {
            Some(strides) => &js_sys::Array::from_iter(strides.into_iter().map(|v| JsValue::from_f64(*v as f64))),
            None => &JsValue::null(),
        };
        map.set(&"strides".to_owned().into(), &strides_obj);

        js_sys::Object::from_entries(&map.into()).unwrap()
    }

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

#[wasm_bindgen]
pub fn ones(shape: Box<[usize]>, dtype: &str) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = to_dtype(dtype)?;
        Ok(DynArray::ones(shape.as_ref(), dtype.inner).into())
    })
}

#[wasm_bindgen]
pub fn zeros(shape: Box<[usize]>, dtype: &str) -> Result<JsArray, String> {
    catch_panic(|| {
        let dtype = to_dtype(dtype)?;
        let arr = DynArray::ones(shape.as_ref(), dtype.inner);
        Ok(arr.into())
    })
}

#[wasm_bindgen]
pub fn from_interchange(obj: JsValue) -> Result<JsArray, String> {
    catch_panic(|| {
        //Ok(ArrayInterchange::try_from_js_value(obj)?.to_array()?.into())
        Ok(serde_wasm_bindgen::from_value::<ArrayInterchange>(obj).map_err(|e| e.to_string())?.to_array()?.into())
    })
}

#[wasm_bindgen(variadic)]
pub fn expr(strs: Vec<String>, lits: Vec<JsArray>) -> Result<JsArray, String> {
    catch_panic(|| {
        //return Err(format!("strs: {:?} lits: {:?}", strs, lits.into_iter().map(|a| a.inner).collect_vec()));
        let expr = parse_with_literals(strs.iter().map(|s| s.as_ref()), lits.into_iter().map(|v| Token::ArrayLit(v.inner)))
            .map_err(|e| format!("{:?}", e))?;
        let vars = HashMap::new();
        match expr.exec(&vars) {
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
    set_panic_hook();
    Ok(())
}

#[wasm_bindgen]
pub fn test() -> String {
    "Hello world!".to_owned()
}
