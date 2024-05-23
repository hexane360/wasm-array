use ndarray::arr2;

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