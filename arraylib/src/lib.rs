#![feature(new_uninit)]
#![feature(trace_macros)]

pub mod bool;
pub mod dtype;
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
