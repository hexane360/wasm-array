use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
fn main() -> Result<(), JsValue> {
    Ok(())
}

#[wasm_bindgen]
pub fn test() -> &'static str {
    return "Hello world!"
}
