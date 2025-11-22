use ndarray::{Array, arr1, arr2};

use arraylib::array::DynArray;
use arraylib::dtype::{DataType, Complex};

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
fn test_matmul() {
    let lhs: DynArray = Array::from_shape_vec(vec![1, 2, 3], vec![
        1f32, 2., 5., 1., 2., 5.
    ]).unwrap().into();

    let rhs: DynArray = Array::from_shape_vec(vec![2, 3, 3], vec![
        5f32, 10., 15., 5., 10., 15.,
        5., 10., 15., 5., 10., 15.,
        5., 10., 15., 5., 10., 15.,
    ]).unwrap().into();

    let out = lhs.mat_mul(rhs);
    assert_eq!(vec![1, 2, 2, 3], out.shape());
    assert_eq!(out.dtype(), DataType::Float32);
    assert_eq!(
        out.downcast::<f32>().unwrap().into_raw_vec_and_offset().0,
        &[40.0f32, 80.0, 120.0, 40.0, 80.0, 120.0, 40.0, 80.0, 120.0, 40.0, 80.0, 120.0],
    )
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

#[test]
fn test_maximum() {
    let a1: DynArray = arr1(&[10u64, 20, 30]).into();
    let a2: DynArray = arr1(&[50u64, 0, 40]).into();
    let expected: DynArray = arr1(&[50u64, 20, 40]).into();

    assert!(expected.allclose(a1.maximum(a2), 1e-8, 0.0));

    let a1: DynArray = arr1(&[10.0, f32::NAN, f32::NAN,    100.0]).into();
    let a2: DynArray = arr1(&[50.0,      0.0, f32::NAN, f32::NAN]).into();
    let expected: DynArray = arr1(&[50.0, f32::NAN, f32::NAN, f32::NAN]).into();

    assert!(expected.allclose(a1.maximum(a2), 1e-8, 0.0));
}

#[test]
fn test_minimum() {
    let a1: DynArray = arr1(&[10u64, 20, 30]).into();
    let a2: DynArray = arr1(&[50u64, 0, 40]).into();
    let expected: DynArray = arr1(&[10u64, 0, 30]).into();

    assert!(expected.allclose(a1.minimum(a2), 1e-8, 0.0));

    let a1: DynArray = arr1(&[10.0, f32::NAN, f32::NAN,    100.0]).into();
    let a2: DynArray = arr1(&[50.0,      0.0, f32::NAN, f32::NAN]).into();
    let expected: DynArray = arr1(&[10.0, f32::NAN, f32::NAN, f32::NAN]).into();

    assert!(expected.allclose(a1.minimum(a2), 1e-8, 0.0));
}

#[test]
fn test_nanmaximum() {
    let a1: DynArray = arr1(&[10.0, f32::NAN, f32::NAN,    100.0]).into();
    let a2: DynArray = arr1(&[50.0,      0.0, f32::NAN, f32::NAN]).into();
    let expected: DynArray = arr1(&[50.0, 0.0, f32::NAN, 100.0]).into();

    assert!(expected.allclose(a1.nanmaximum(a2), 1e-8, 0.0));
}

#[test]
fn test_nanminimum() {
    let a1: DynArray = arr1(&[10.0, f32::NAN, f32::NAN,    100.0]).into();
    let a2: DynArray = arr1(&[50.0,     00.0, f32::NAN, f32::NAN]).into();
    let expected: DynArray = arr1(&[10.0, 0.0, f32::NAN, 100.0]).into();

    assert!(expected.allclose(a1.nanminimum(a2), 1e-8, 0.0));
}

#[test]
fn test_exp() {
    let a1: DynArray = arr1(&[1.0f32, 2.0, -1.0, -5.0]).into();
    let expected: DynArray = arr1(&[2.7182817, 7.389056, 0.36787945, 0.006737947]).into();

    //println!("actual: {}", a1.exp());
    assert!(expected.allclose(a1.exp(), 1e-6, 1e-8));
}

#[test]
fn test_exp_complex() {
    let a1: DynArray = arr1(&[Complex::new(0.0f64, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 3.141592653589793), Complex::new(0.0, -3.141592653589793)]).into();
    let expected: DynArray = arr1(&[Complex::new(1.0f64, 0.0), Complex::new(2.718281828459045f64, 0.0), Complex::new(-1.0f64, 0.0), Complex::new(-1.0f64, 0.0)]).into();

    //println!("actual: {}", a1.exp());
    assert!(expected.allclose(a1.exp(), 1e-12, 1e-12));
}