#![feature(new_uninit)]
#![feature(type_alias_impl_trait)]
#![feature(convert_float_to_int)]

pub mod bool;
pub mod dtype;
mod cast;
//pub mod typedarray;
pub mod array;


pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
