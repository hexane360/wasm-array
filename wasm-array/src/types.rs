use std::borrow::Cow;
use std::mem;
use core::fmt;

use arraylib::bool::Bool;
use ndarray::{Array1, ArrayViewD};
use serde::{Serialize, Deserialize, de, ser};
use wasm_bindgen::prelude::*;
use base64::engine::{Engine, general_purpose::URL_SAFE as BASE64_URL_SAFE};

use arraylib::array::DynArray;
use arraylib::dtype::{DataType, DataTypeCategory, PhysicalType};
use arraylib::arraylike::{NestedList, ArrayValue};
use arraylib::colors::named_colors;

use super::{
    IArrayInterchange, ColorLike, DataTypeLike,
    ShapeLike, JsDataType, JsArray,
};

/*
fn jsval_as_string(val: &JsValue) -> String {
    js_sys::Reflect::get(val, &JsValue::from("toString")).ok()
        .and_then(|prop| prop.dyn_into::<js_sys::Function>().ok())
        .and_then(|f| f.apply(val, &js_sys::Array::of1(&JsValue::from_f64(10.))).ok())
        .and_then(|s| s.as_string())
        .unwrap_or_else(|| "<error printing JS value>".to_owned())
}
*/

fn jsval_type_as_string(val: &JsValue) -> String {
    val.js_typeof().as_string().unwrap_or_else(|| "<error getting JS type>".to_owned())
}

pub trait DowncastWasmExport: wasm_bindgen::convert::RefFromWasmAbi<Abi = u32> {

    fn __get_type_id() -> &'static str;

    fn downcast_ref(js: &JsValue) -> Option<<Self as wasm_bindgen::convert::RefFromWasmAbi>::Anchor> {
        if !js.is_object() { return None; }

        let js_typeid_function = match js_sys::Reflect::get(js, &JsValue::from("__getTypeId")).map(|val| val.dyn_into::<js_sys::Function>()) {
            Ok(Ok(val)) => val,
            _ => return None,
        };
        let js_type_id = match js_sys::Reflect::apply(&js_typeid_function, js, &js_sys::Array::new()).map(|val| val.as_string()) {
            Ok(Some(val)) => val,
            _ => return None,
        };

        if js_type_id != Self::__get_type_id() { return None; }

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

#[wasm_bindgen(js_class = DataType)]
impl JsDataType {
    #[wasm_bindgen(js_name = "__getTypeId")]
    pub fn __js_get_type_id(&self) -> String {
        "HTxHNbDq84KHGdf5".to_owned()
    }
}

impl DowncastWasmExport for JsDataType {
    fn __get_type_id() -> &'static str { "HTxHNbDq84KHGdf5" }
}

#[wasm_bindgen(js_class = NArray)]
impl JsArray {
    #[wasm_bindgen(js_name = "__getTypeId")]
    pub fn __js_get_type_id(&self) -> String {
        "dbvjUb9K4VrV2Fq5".to_owned()
    }
}

impl DowncastWasmExport for JsArray {
    fn __get_type_id() -> &'static str { "dbvjUb9K4VrV2Fq5" }
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
        let dtype = self.obj.as_string().ok_or_else(|| format!(
            "Invalid datatype. Expected a DataType object or string, instead got object {:?} of type {:?}",
            self.obj, jsval_type_as_string(&self.obj))
        )?;
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

struct BytesOrBase64Visitor;

impl<'de> de::Visitor<'de> for BytesOrBase64Visitor {
    type Value = Box<[u8]>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("ArrayBuffer, TypedArray, or base64-encoded string")
    }

    fn visit_seq<V: de::SeqAccess<'de>>(self, mut visitor: V) -> Result<Self::Value, V::Error> {
        let len = std::cmp::min(visitor.size_hint().unwrap_or(0), 4096);
        let mut bytes = Vec::with_capacity(len);

        while let Some(b) = visitor.next_element()? {
            bytes.push(b);
        }

        Ok(bytes.into_boxed_slice())
    }

    fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Ok(v.to_owned().into_boxed_slice())
    }

    fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
        Ok(Box::from(v))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        BASE64_URL_SAFE.decode(v).map(|v| v.into_boxed_slice()).map_err(|e| E::custom(e.to_string()))
    }

    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        self.visit_str(&v)
    }
}

pub fn deserialize_bytes_or_base64<'de, D>(deserializer: D) -> Result<Box<[u8]>, D::Error>
where D: de::Deserializer<'de>
{
    deserializer.deserialize_any(BytesOrBase64Visitor)
    //deserializer.deserialize_byte_buf(BytesOrBase64Visitor)
}

pub fn serialize_bytes<S>(val: &[u8], serializer: S) -> Result<S::Ok, S::Error>
where S: ser::Serializer
{
    serializer.serialize_bytes(val)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArrayInterchange {
    #[serde(rename = "data", with = "serde_wasm_bindgen::preserve")]
    buf: JsValue,
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

fn parse_buffer_or_array(value: &JsValue) -> Result<Box<[u8]>, String> {
    if let Some(buf) = value.as_string() {
        BASE64_URL_SAFE.decode(buf.as_bytes()).map(|v| v.into_boxed_slice())
            .map_err(|e| format!("Invalid array data, expected a base64-encoded string: {}", e))
    } else if let Some(buf) = value.dyn_ref::<js_sys::ArrayBuffer>() {
        Ok(js_sys::Uint8Array::new(buf).to_vec().into_boxed_slice())
    } else if let Some(arr) = value.dyn_ref::<js_sys::Array>() {
        serde_wasm_bindgen::from_value::<Box<[u8]>>(arr.into()).map_err(|e| e.to_string())
    } else if let Some(buf) = js_sys::Reflect::get(&value, &JsValue::from("buffer")).ok()
    .and_then(|v| v.dyn_into::<js_sys::ArrayBuffer>().ok()) {
        Ok(js_sys::Uint8Array::new(&buf).to_vec().into_boxed_slice())
    } else {
        Err(format!("Invalid array data: '{:?}', expected an ArrayBuffer, TypedArray, or base64-encoded string", value))
    }
}

impl ArrayInterchange {
    pub fn to_array(self) -> Result<DynArray, String> {
        DynArray::from_buf(parse_buffer_or_array(&self.buf)?, self.dtype, self.shape, self.strides)
    }
}

impl From<DynArray> for ArrayInterchange {
    fn from(value: DynArray) -> Self {
        let (buf, dtype, shape, strides) = value.to_buf();

        Self {
            buf: js_sys::Uint8Array::from(buf.as_ref()).into(), dtype, shape, strides, version: 3,
        }
    }
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

pub enum JsCow<'a, T, U: AsRef<T>> {
    Owned(T),
    Borrowed(&'a T),
    WasmRef(wasm_bindgen::__rt::Ref<'a, U>),
}

pub fn parse_array<'a>(arr: &'a JsValue, dtype: Option<DataType>) -> Option<Cow<'a, DynArray>> {
    JsArray::downcast_ref(&arr).map(|val| {
        // SAFETY: `arr` is a JsArray borrowed for 'a, ensuring the underlying
        // array won't be dropped or mutated for at least that region
        let array_ref: &'a DynArray = unsafe { mem::transmute(&val.inner) };
        //mem::forget(val);
        //log::log(format!("parse_array got existing array {:?}, casting to {:?}", val.as_ref(), &dtype));
        match dtype {
            Some(dtype) => Cow::Owned(array_ref.cast(dtype).into_owned()),
            None => Cow::Borrowed(array_ref),
        }
    })
}

pub fn parse_arraylike<'a>(arr: &'a JsValue, dtype: Option<DataType>) -> Result<Cow<'a, DynArray>, String> {
    //log::log(format!("parse_arraylike({:?})", arr));
    if let Some(arr) = parse_array(arr, dtype) {
        return Ok(arr);
    }

    parse_nestedlist(arr)?.build_array(dtype).map(Cow::Owned)
    //Err(format!("Array conversion not implemented for type '{}'", arr.js_typeof().as_string().unwrap()))
}

fn parse_nestedlist(val: &JsValue) -> Result<NestedList, String> {
    if val.is_object() {
        if let Some(it) = js_sys::try_iter(val).map_err(|e| format!("Error in JS: {:?}", e))? {
            return Ok(NestedList::Array(
                it.map(|val|
                    val.map_err(|e| format!("JS error in iterator: {:?}", e)).and_then(|val| parse_nestedlist(&val))
                ).try_collect()?
            ));
        }
    }
    parse_arrayvalue(val).map(|val| NestedList::Value(val))
}

fn parse_arrayvalue(val: &JsValue) -> Result<ArrayValue, String> {
    if let Some(val) = val.as_f64() {
        return Ok({
            if val == 0. || (val.round() - val).abs() <= 1e-6 * val.abs() && val.is_finite() {
                ArrayValue::Int(val.round() as i64)
            } else {
                ArrayValue::Float(val)
            }
        });
    }

    if let Some(arr) = parse_array(val, None) {
        return ArrayValue::from_arr(arr.as_ref())
            .ok_or_else(|| format!("Expected number or scalar array, instead got array of shape {:?}", arr.shape()));
    }

    if let Some(val) = val.as_bool() {
        return Ok(ArrayValue::Boolean(val.into()));
    }

    if let Ok(val) = val.clone().dyn_into::<js_sys::BigInt>() {
        return i64::try_from(val.clone()).map(ArrayValue::Int)
            .map_err(|_| format!("BigInt '{}' overflows i64", val));
    }

    Err(format!("Array conversion not implemented for type '{}'", val.js_typeof().as_string().unwrap()))
    //Err(format!("Unexpected type '{}'. Expected a number.", val.js_typeof().as_string().unwrap()))
}

fn to_nested_array_inner<T: PhysicalType, F>(arr: ArrayViewD<'_, T>, f: F) -> JsValue
where F: Fn(T) -> JsValue + Copy
{
    if arr.ndim() == 0 {
        f(*arr.iter().next().unwrap())
    } else {
        let arr: js_sys::Array = arr.outer_iter().map(|inner| to_nested_array_inner(inner, f)).collect();
        arr.into()
    }
}

pub fn to_nested_array(arr: &DynArray) -> Result<JsValue, String> {
    match arr.dtype().category() {
        DataTypeCategory::Complex => Err(format!("toNestedArray unsuported for type {}", arr.dtype())),
        DataTypeCategory::Boolean => Ok(to_nested_array_inner(arr.downcast_ref::<Bool>().unwrap().view(), |val| JsValue::from_bool(val.into()))),
        _ => {
            let arr = arr.cast(DataType::Float64);
            Ok(to_nested_array_inner(arr.downcast_ref::<f64>().unwrap().view(), JsValue::from_f64))
        }
    }
}
