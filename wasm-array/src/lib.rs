use std::collections::HashMap;

use wasm_bindgen::prelude::*;

pub mod expr;

use expr::{parse_with_literals, Token};

use arraylib::array::DynArray;
use arraylib::dtype::DataType;

#[wasm_bindgen(js_name = DataType)]
pub struct JsDataType {
    inner: DataType
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

#[wasm_bindgen]
pub fn ones(shape: Box<[usize]>, dtype: &str) -> Result<JsArray, String> {
    let dtype = to_dtype(dtype)?;
    Ok(DynArray::ones(shape.as_ref(), dtype.inner).into())
}

#[wasm_bindgen(variadic)]
pub fn expr(strs: Vec<String>, lits: Vec<JsArray>) -> Result<JsArray, String> {
    //return Err(format!("strs: {:?} lits: {:?}", strs, lits.into_iter().map(|a| a.inner).collect_vec()));
    let expr = parse_with_literals(strs.iter().map(|s| s.as_ref()), lits.into_iter().map(|v| Token::ArrayLit(v.inner)))
        .map_err(|e| format!("{:?}", e))?;
    let vars = HashMap::new();
    match expr.exec(&vars) {
        Ok(arr) => Ok(arr.into()),
        Err(e) => Err(format!("{:?}", e)),
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
