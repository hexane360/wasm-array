use std::borrow::Borrow;

use ndarray::{Array, ArrayView, ArrayView1, Axis, IxDyn};

use arraylib_macro::type_dispatch;
use crate::dtype::{Complex, DataTypeCategory, PhysicalType};
use crate::array::DynArray;
use crate::util::normalize_axis;

/// Flattens an array view
/// Requires that the view is contiguous (a default view of an owned array should be).
/// The flattened view will be in memory order, so this should be used with functions
/// where ordering doesn't matter.
fn flatten_view<'a, T>(arr: ArrayView<'a, T, IxDyn>) -> ArrayView1<'a, T> {
    let slice = arr.to_slice_memory_order().unwrap();
    ArrayView1::from_shape((slice.len(),), slice).unwrap()
}

fn reduce_along_axes<'a, T: Copy, F: Fn(T, T) -> T + Copy>(arr: ArrayView<'a, T, IxDyn>, mut axes: Vec<usize>, f: F) -> Result<Array<T, IxDyn>, String> {
    axes.sort_by_key(|v| usize::MAX - v);

    if axes.len() == 0 {
        return Ok(arr.to_owned())
    }

    let closure = |ax: ArrayView1<'_, T>| ax.iter().copied().reduce(f).unwrap();

    let inner_fn = |arr: ArrayView<'_, T, IxDyn>, ax: usize| -> Result<Array<T, IxDyn>, String> {
        if arr.shape()[ax] == 0 {
            Err(format!("Reduction along a zero-sized axis"))
        } else {
            Ok(arr.map_axis(Axis(ax), &closure))
        }
    };

    let mut it = axes.iter();

    let mut axes_owned = inner_fn(arr, *it.next().unwrap())?;

    for &ax in it {
        axes_owned = inner_fn(axes_owned.view(), ax)?;
    }
    Ok(axes_owned)
}

fn filter_reduce_along_axes<'a, T: Copy, P: Fn(T) -> bool + Copy, F: Fn(T, T) -> T + Copy>(arr: ArrayView<'a, T, IxDyn>, mut axes: Vec<usize>, f: F, pred: P) -> Result<Array<T, IxDyn>, String> {
    axes.sort_by_key(|v| usize::MAX - v);

    let inner_fn = |arr: ArrayView<'_, T, IxDyn>, ax: usize| -> Result<Array<T, IxDyn>, String> {
        let opt_arr = arr.fold_axis(Axis(ax), None, |&accum, &val| match (accum, if pred(val) { Some(val) } else { None }) {
            (None, val) => val,
            (Some(prev), None) => Some(prev),
            (Some(prev), Some(val)) => Some(f(prev, val)),
        });
        let shape = opt_arr.shape().to_owned();
        Ok(Array::from_shape_vec(shape, opt_arr.into_raw_vec().into_iter()
            .try_collect().ok_or_else(|| format!("Reduction along a zero-sized axis"))?
        ).unwrap())
    };

    let mut it = axes.iter();

    let mut axes_owned = inner_fn(arr, *it.next().unwrap())?;

    for &ax in it {
        axes_owned = inner_fn(axes_owned.view(), ax)?;
    }
    Ok(axes_owned)
}

macro_rules! impl_reduction {
    ( $arr:expr, $axes:expr, $tys:ty, $fn:expr ) => { {
        let arr = $arr;
        match $axes {
            None => {
                type_dispatch!(
                    $tys,
                    |ref arr| { reduce_along_axes(flatten_view(arr.view()).into_dyn(), vec![0], $fn).map(|arr| arr.into()) }
                )
            },
            Some(axes) => {
                let axes: Vec<usize> = axes.iter().map(|&ax| normalize_axis(ax, arr.ndim())).collect();
                type_dispatch!(
                    $tys,
                    |ref arr| { reduce_along_axes(arr.view(), axes, $fn).map(|arr| arr.into()) }
                )
            }
        }
    } }
}

macro_rules! impl_filter_reduction {
    ( $arr:expr, $axes:expr, $tys:ty, $fn:expr, $pred:expr ) => { {
        let arr = $arr;
        match $axes {
            None => {
                type_dispatch!(
                    $tys,
                    |ref arr| { filter_reduce_along_axes(flatten_view(arr.view()).into_dyn(), vec![0], $fn, $pred).map(|arr| arr.into()) }
                )
            },
            Some(axes) => {
                let axes: Vec<usize> = axes.iter().map(|&ax| normalize_axis(ax, arr.ndim())).collect();
                type_dispatch!(
                    $tys,
                    |ref arr| { filter_reduce_along_axes(arr.view(), axes, $fn, $pred).map(|arr| arr.into()) }
                )
            }
        }
    } }
}

pub fn min<A: Borrow<DynArray>>(arr: A, axes: Option<&[isize]>) -> Result<DynArray, String> {
    let arr = arr.borrow();
    impl_reduction!(arr, axes, (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64), |l, r| l.min(r))
}

pub fn max<A: Borrow<DynArray>>(arr: A, axes: Option<&[isize]>) -> Result<DynArray, String> {
    let arr = arr.borrow();
    impl_reduction!(arr, axes, (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64), |l, r| l.max(r))
}

pub fn sum<A: Borrow<DynArray>>(arr: A, axes: Option<&[isize]>) -> Result<DynArray, String> {
    let arr = arr.borrow();
    impl_reduction!(arr, axes, (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>), |l, r| l + r)
}

pub fn prod<A: Borrow<DynArray>>(arr: A, axes: Option<&[isize]>) -> Result<DynArray, String> {
    let arr = arr.borrow();
    impl_reduction!(arr, axes, (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>), |l, r| l * r)
}

pub fn nanmin<A: Borrow<DynArray>>(arr: A, axes: Option<&[isize]>) -> Result<DynArray, String> {
    let arr = arr.borrow();
    if arr.dtype().category() < DataTypeCategory::Floating {
        return min(arr, axes);
    }

    impl_filter_reduction!(arr, axes, (f32, f64), |l, r| l.min(r), |v| !v.is_nan())
}

pub fn nanmax<A: Borrow<DynArray>>(arr: A, axes: Option<&[isize]>) -> Result<DynArray, String> {
    let arr = arr.borrow();
    if arr.dtype().category() < DataTypeCategory::Floating {
        return max(arr, axes);
    }

    impl_filter_reduction!(arr, axes, (f32, f64), |l, r| l.max(r), |v| !v.is_nan())
}