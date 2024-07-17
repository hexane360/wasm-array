use std::borrow::Borrow;
use std::sync::{LazyLock, RwLock};

use num::{Float, Zero, ToPrimitive};
use ndarray::{Array1, ArrayViewMut, Axis, IxDyn, Dimension};
use rustfft::{FftNum, FftPlanner, Fft};

use arraylib_macro::type_dispatch;
use crate::dtype::{DataType, DataTypeCategory, Complex};
use crate::array::{DynArray, roll_inner};
use crate::util::normalize_axis;

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FFTNorm {
    #[default]
    Backwards,
    Forwards,
    Ortho,
}

impl Into<&'static str> for FFTNorm {
    fn into(self) -> &'static str {
        match self {
            FFTNorm::Backwards => "backward",
            FFTNorm::Forwards => "forward",
            FFTNorm::Ortho => "ortho",
        }
    }
}

impl<'a> TryFrom<&'a str> for FFTNorm {
    type Error = String;
    
    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        Ok(match value.to_lowercase().as_ref() {
            "backward" => FFTNorm::Backwards,
            "forward" => FFTNorm::Forwards,
            "ortho" => FFTNorm::Ortho,
            _ => return Err(format!("Unknown normalization '{}'", value)),
        })
    }
}

static FFT_PLANNER_64: LazyLock<RwLock<rustfft::FftPlanner<f32>>> = LazyLock::new(|| RwLock::new(FftPlanner::new()));
static FFT_PLANNER_128: LazyLock<RwLock<rustfft::FftPlanner<f64>>> = LazyLock::new(|| RwLock::new(FftPlanner::new()));

fn fft_inplace<T>(mut arr: ArrayViewMut<Complex<T>, IxDyn>, axis: usize, plan: &dyn Fft<T>, norm_scale: Option<T>)
where T: FftNum + Float,
{
    let n = arr.shape()[axis];
    debug_assert_eq!(n, plan.len());
    let mut scratch: Vec<Complex<T>> = vec![Complex::zero(); plan.get_inplace_scratch_len()];

    if arr.is_standard_layout() {
        let inner_axis = arr.ndim() - 1;
        if axis == inner_axis {
            let rows = arr.as_slice_mut().unwrap();
            plan.process_with_scratch(rows, &mut scratch);

            if let Some(scale) = norm_scale {
                apply_scaling(arr.view_mut(), scale);
            }
            /*for mut row in arr.rows_mut() {
                let row = row.as_slice_mut().unwrap();
                plan.process_with_scratch(row, &mut scratch);
            }*/
            return
        }
    }
    // otherwise we use a temporary buffer
    // TODO batch along this dimension
    let mut buf: Array1<Complex<T>> = Array1::zeros(n);
    for mut lane in arr.lanes_mut(Axis(axis)) {
        buf.assign(&lane);
        plan.process_with_scratch(buf.as_slice_mut().unwrap(), &mut scratch);
        lane.assign(&buf);
    }

    if let Some(scale) = norm_scale {
        apply_scaling(arr.view_mut(), scale)
    }
}

fn apply_scaling<'a, T, D>(mut arr: ArrayViewMut<'a, Complex<T>, D>, scale: T)
    where T: Float, D: Dimension
{
    arr.mapv_inplace(|v| Complex::new(v.re * scale, v.im * scale))
}

pub fn fft<A: Borrow<DynArray>>(arr: A, axes: &[isize], norm: Option<FFTNorm>) -> DynArray {
    let mut s = arr.borrow().cast_category(DataTypeCategory::Complex).into_owned();
    let (shape, ndim) = (s.shape(), s.ndim());
    let norm = norm.unwrap_or_default();

    for axis in axes.into_iter().map(|axis| normalize_axis(*axis, ndim)) {
        let len = shape[axis];
        match s.dtype() {
            DataType::Complex64 => {
                let plan = {
                    let mut planner = FFT_PLANNER_64.write().unwrap();
                    planner.plan_fft_forward(len)
                };

                let norm_scale = match norm {
                    FFTNorm::Backwards => None,
                    FFTNorm::Forwards => Some(1f32 / len.to_f32().unwrap()),
                    FFTNorm::Ortho => Some(1f32 / len.to_f32().unwrap().sqrt()),
                };

                fft_inplace(s.downcast_mut::<Complex<f32>>().unwrap().view_mut(), axis, plan.as_ref(), norm_scale);
            },
            DataType::Complex128 => {
                let plan = {
                    let mut planner = FFT_PLANNER_128.write().unwrap();
                    planner.plan_fft_forward(len)
                };

                let norm_scale = match norm {
                    FFTNorm::Backwards => None,
                    FFTNorm::Forwards => Some(1f64 / len.to_f64().unwrap()),
                    FFTNorm::Ortho => Some(1f64 / len.to_f64().unwrap().sqrt()),
                };

                fft_inplace(s.downcast_mut::<Complex<f64>>().unwrap().view_mut(), axis, plan.as_ref(), norm_scale);
            },
            _ => unreachable!(),
        }
    }
    s
}

pub fn ifft<A: Borrow<DynArray>>(arr: A, axes: &[isize], norm: Option<FFTNorm>) -> DynArray {
    let mut s = arr.borrow().cast_category(DataTypeCategory::Complex).into_owned();
    let (shape, ndim) = (s.shape(), s.ndim());
    let norm = norm.unwrap_or_default();

    for axis in axes.into_iter().map(|axis| normalize_axis(*axis, ndim)) {
        let len = shape[axis];
        match s.dtype() {
            DataType::Complex64 => {
                let plan = {
                    let mut planner = FFT_PLANNER_64.write().unwrap();
                    planner.plan_fft_inverse(len)
                };

                let norm_scale = match norm {
                    FFTNorm::Forwards => None,
                    FFTNorm::Backwards => Some(1f32 / len.to_f32().unwrap()),
                    FFTNorm::Ortho => Some(1f32 / len.to_f32().unwrap().sqrt()),
                };

                fft_inplace(s.downcast_mut::<Complex<f32>>().unwrap().view_mut(), axis, plan.as_ref(), norm_scale);
            },
            DataType::Complex128 => {
                let plan = {
                    let mut planner = FFT_PLANNER_128.write().unwrap();
                    planner.plan_fft_inverse(len)
                };

                let norm_scale = match norm {
                    FFTNorm::Forwards => None,
                    FFTNorm::Backwards => Some(1f64 / len.to_f64().unwrap()),
                    FFTNorm::Ortho => Some(1f64 / len.to_f64().unwrap().sqrt()),
                };

                fft_inplace(s.downcast_mut::<Complex<f64>>().unwrap().view_mut(), axis, plan.as_ref(), norm_scale);
            },
            _ => unreachable!(),
        }
    }
    s
}

pub fn fftshift<A: Borrow<DynArray>>(arr: A, axes: &[isize]) -> DynArray {
    let arr = arr.borrow();

    let axes: Vec<usize> = axes.iter().map(|ax| normalize_axis(*ax, arr.ndim())).collect();

    let mut ax_rolls: Vec<isize> = vec![0; arr.ndim()];
    for (ax, size) in axes.into_iter().zip(arr.shape()) {
        ax_rolls[ax] = (size >> 1) as isize;
    }

    type_dispatch!(
        (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
        |ref arr| roll_inner(arr, &ax_rolls).into()
    )
}

pub fn ifftshift<A: Borrow<DynArray>>(arr: A, axes: &[isize]) -> DynArray {
    let arr = arr.borrow();

    let axes: Vec<usize> = axes.iter().map(|ax| normalize_axis(*ax, arr.ndim())).collect();

    let mut ax_rolls: Vec<isize> = vec![0; arr.ndim()];
    for (ax, size) in axes.into_iter().zip(arr.shape()) {
        ax_rolls[ax] = -((size >> 1) as isize);
    }

    type_dispatch!(
        (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, Complex<f32>, Complex<f64>),
        |ref arr| roll_inner(arr, &ax_rolls).into()
    )
}