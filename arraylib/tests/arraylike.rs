
use ndarray::{arr1, arr2};

use arraylib::dtype::{Complex, DataType};
use arraylib::arraylike::{ArrayValue, NestedList};

#[test]
fn test_arraylike() {
    let test1 = NestedList::Array(vec![
        NestedList::Array(vec![
            NestedList::Value(ArrayValue::Float(3.2)),
            NestedList::Value(ArrayValue::Float(8.2)),
        ]),
        NestedList::Array(vec![
            NestedList::Value(ArrayValue::Int(-3)),
            NestedList::Value(ArrayValue::Int(-5)), 
        ]),
    ]);

    assert_eq!(
        test1.build_array(None),
        Ok(arr2(&[[3.2f64, 8.2], [-3., -5.]]).into())
    );
    assert_eq!(
        test1.build_array(Some(DataType::Int64)),
        Ok(arr2(&[[3i64, 8], [-3, -5]]).into())
    );
}

#[test]
fn test_arraylike_shape_mismatch() {
    let test1 = NestedList::Array(vec![
        NestedList::Array(vec![
            NestedList::Value(ArrayValue::Float(3.2)),
            NestedList::Value(ArrayValue::Float(8.2)),
        ]),
        NestedList::Array(vec![
            NestedList::Value(ArrayValue::Int(-3)),
            NestedList::Value(ArrayValue::Int(-5)), 
        ]),
        NestedList::Array(vec![]),
    ]);

    assert_eq!(
        test1.build_array(None),
        Err("Shape mismatch at dim 0: Subarray 0 is of shape [2], but subarray 2 is of shape [0]".to_owned())
    );
}

#[test]
fn test_arraylike_cast_error() {
    let test = NestedList::Array(vec![
        NestedList::Value(ArrayValue::Complex(Complex::I * 5.0f64)),
        NestedList::Value(ArrayValue::Int(5)),
        NestedList::Value(ArrayValue::Int(20)),
    ]);

    assert_eq!(
        test.build_array(None),
        Ok(arr1(&[
            Complex::new(0.0, 5.0f64),
            Complex::new(5.0, 0.0f64),
            Complex::new(20.0, 0.0f64),
        ]).into()),
    );

    assert_eq!(
        test.build_array(Some(DataType::Float64)),
        Err("Unable to build array of dtype 'float64' from value of type 'complex'".to_owned()),
    );
}