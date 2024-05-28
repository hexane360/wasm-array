use ndarray::{arr1, arr2};

use arraylib::array::DynArray;

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