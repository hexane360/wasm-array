
use rstest::rstest;

use arraylib::dtype::{DataType, DataTypeOrCategory, DataTypeCategory, promote_types};

#[rstest]
// basic dtypes promotion
#[case(&[DataType::Boolean, DataType::Boolean], DataType::Boolean)]
#[case(&[DataType::UInt8, DataType::UInt32], DataType::UInt32)]
#[case(&[DataType::Int8, DataType::Int16], DataType::Int16)]
#[case(&[DataType::Float32, DataType::Float64], DataType::Float64)]
#[case(&[DataType::Complex64, DataType::Complex128], DataType::Complex128)]
// unsigned -> signed conversion
#[case(&[DataType::UInt8, DataType::Int8], DataType::Int16)]
#[case(&[DataType::UInt16, DataType::Int8], DataType::Int32)]
#[case(&[DataType::UInt32, DataType::Int8], DataType::Int64)]
#[case(&[DataType::UInt64, DataType::Int8], DataType::Int64)]
// int -> float conversion
#[case(&[DataType::UInt16, DataType::Float32], DataType::Float32)]
#[case(&[DataType::Int32, DataType::Float32], DataType::Float64)]
// float -> complex conversion
#[case(&[DataType::Float32, DataType::Complex64], DataType::Complex64)]
#[case(&[DataType::Float64, DataType::Complex64], DataType::Complex128)]
fn test_promote_types(#[case] input: &'static [DataType], #[case] expected: DataType) {
    assert_eq!(promote_types(input), expected);

    // check commutativity
    let mut reversed: Vec<_> = input.into_iter().rev().copied().collect();
    assert_eq!(promote_types(&reversed), expected);

    // and idempotence
    reversed.push(expected);
    assert_eq!(promote_types(&reversed), expected);
}

#[test]
fn test_promote_non_associative() {
    // Int16, UInt16 => UInt32
    // Float32, UInt32 => Float64
    assert_eq!(
        promote_types(&[DataType::Float32, promote_types(&[DataType::Int16, DataType::UInt16])]),
        DataType::Float64,
    );

    // Float32, UInt16 => Float32
    // Int16, Float32 => Float32
    assert_eq!(
        promote_types(&[DataType::Int16, promote_types(&[DataType::Float32, DataType::UInt16])]),
        DataType::Float32,
    );
}

#[rstest]
#[case(&[DataTypeOrCategory::DataType(DataType::Int8), DataTypeOrCategory::Category(DataTypeCategory::Signed)], DataType::Int8)]
#[case(&[DataTypeOrCategory::Category(DataTypeCategory::Signed)], DataType::Int64)]
#[case(&[DataTypeOrCategory::Category(DataTypeCategory::Complex), DataTypeOrCategory::DataType(DataType::Float32)], DataType::Complex64)]
#[case(&[DataTypeOrCategory::Category(DataTypeCategory::Complex), DataTypeOrCategory::Category(DataTypeCategory::Floating)], DataType::Complex128)]
fn test_promote_generic(#[case] input: &'static [DataTypeOrCategory], #[case] expected: DataType) {
    assert_eq!(promote_types(input), expected);

    // check commutativity
    let reversed: Vec<_> = input.into_iter().rev().copied().collect();
    assert_eq!(promote_types(&reversed), expected);
}