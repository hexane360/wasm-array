use ndarray::{arr0, arr1, arr2};

use arraylib::array::DynArray;
use arraylib::reductions::{min, nanmin};

#[test]
fn test_reduce_min() {
    let a1: DynArray = arr2(&[[20u8, 50u8, 100u8],
                              [10u8, 50u8, 200u8]]).into();

    assert_eq!(min(&a1, None), Ok(arr0(10u8).into()));
    assert_eq!(min(&a1, Some(&[0])), Ok(arr1(&[10u8, 50, 100]).into()));

    let slice: &[u8] = &[];
    let empty: DynArray = arr1(slice).into();

    assert_eq!(min(&empty, None), Err("Reduction along a zero-sized axis".to_owned()));
    assert_eq!(min(&empty, Some(&[0])), Err("Reduction along a zero-sized axis".to_owned()));
}

#[test]
fn test_reduce_nanmin() {
    let a1: DynArray = arr2(&[[20., 50., 100.],
                              [10., f64::NAN, 200.]]).into();

    assert_eq!(nanmin(&a1, None), Ok(arr0(10f64).into()));
    assert_eq!(nanmin(&a1, Some(&[0])), Ok(arr1(&[10f64, 50., 100.]).into()));

    let a2: DynArray = arr2(&[[20., f64::NAN, 100.],
                              [10., f64::NAN, 200.]]).into(); 

    use arraylib::dtype::IsClose;
    assert!(f64::NAN.is_close(f64::NAN, 1e-8, 0.));

    assert_eq!(nanmin(&a2, None), Ok(arr0(10f64).into()));
    println!("{}", nanmin(&a2, Some(&[0])).unwrap());
    assert!(nanmin(&a2, Some(&[0])).unwrap().allclose(&arr1(&[10., f64::NAN, 100.]).into(), 1e-8, 0.));
    assert_eq!(nanmin(&a2, Some(&[1])), Ok(arr1(&[20f64, 10.]).into()));
}