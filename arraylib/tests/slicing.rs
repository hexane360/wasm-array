use ndarray::{arr1, arr2};

use arraylib::array::DynArray;

#[test]
fn test_roll_1d() {
    let arr: DynArray = arr1(&[1i16, 2, 3, 4, 5]).into();

    let expected = arr1(&[2i16, 3, 4, 5, 1]).into();
    assert_eq!(
        arr.roll(&[-1], &[0]),
        expected
    );

    let expected = arr1(&[5i16, 1, 2, 3, 4]).into();
    assert_eq!(
        arr.roll(&[1], &[0]),
        expected
    );
}

#[test]
fn test_roll_2d() {
    let arr: DynArray = arr2(&[
        [1i16, 2, 3, 4],
        [5i16, 6, 7, 8],
        [9i16, 10, 11, 12],
    ]).into();

    let expected: DynArray = arr2(&[
        [ 4i16, 1,  2,  3],
        [ 8i16, 5,  6,  7],
        [12i16, 9, 10, 11],
    ]).into();
    assert_eq!(
        arr.roll(&[1], &[-1]),
        expected
    );

    let expected: DynArray = arr2(&[
        [ 6i16,  7,  8, 5],
        [10i16, 11, 12, 9],
        [ 2i16,  3,  4, 1],
    ]).into();
    assert_eq!(
        arr.roll(&[3, 2], &[1, 0]),
        expected
    );
}

#[test]
fn test_reshape() {
    let arr: DynArray = arr1(&[1i16, 2, 3, 4, 5, 6]).into();

    let expected: Result<DynArray, String> = Ok(arr2(&[
        [1i16, 2, 3],
        [4i16, 5, 6],
    ]).into());
    assert_eq!(&arr.reshape(&[2, 3]), &expected);
    assert_eq!(&arr.reshape(&[-1, 3]), &expected);
    assert_eq!(&arr.reshape(&[2, -1]), &expected);

    assert_eq!(arr.reshape(&[-1, -1]), Err("Cannot use -1 on multiple dimensions in shape [-1, -1]".to_owned()));
    assert_eq!(arr.reshape(&[1, 1]), Err("Cannot reshape array of size 6 into shape [1, 1]".to_owned()));
}