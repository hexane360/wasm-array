use ndarray::{arr1, arr2};

use arraylib::array::DynArray;
use arraylib::dtype::DataType;

#[test]
fn test_add_numeric() {
    let a1: DynArray = arr2(&[[20u8, 50u8, 100u8],
                              [20u8, 50u8, 200u8]]).into();

    let a2: DynArray = arr2(&[[20u8, 50u8, 50u8],
                              [20u8, 50u8, 100u8]]).into();

    let expected: DynArray = arr2(&[[40u8, 100u8, 150u8],
                                    [40u8, 100u8, 44u8]]).into();

    assert_eq!(expected, a1 + a2);
}

#[test]
fn test_add_type_promote() {
    let a1: DynArray = arr1(&[20i8, 50i8, 100i8]).into();
    let a2: DynArray = arr1(&[20u32, 50u32, 100u32]).into();

    let expected: DynArray = arr1(&[40i64, 100i64, 200i64]).into();
    assert_eq!(expected, a1 + a2);
}

#[test]
fn test_from_buf() {
    let a: DynArray = DynArray::from_buf(
        vec![1, 2, 3, 4].into_boxed_slice(),
        DataType::UInt8,
        vec![1, 4].into_boxed_slice(),
        Some(vec![0, 1].into_boxed_slice()),
    ).unwrap();

    let expected: DynArray = arr2(&[[1u8, 2u8, 3u8, 4u8]]).into();

    assert_eq!(expected, a);
}