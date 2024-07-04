
use std::sync::OnceLock;

use num::{Float, ToPrimitive};
use ndarray::{ArrayViewD, ArrayView2, ArrayD, Zip, Axis, s};
use include_bytes_aligned::include_bytes_aligned;

fn color_data_as_arr(bytes: &'static [u8]) -> ArrayView2<'static, f32> {
    ArrayView2::from_shape((bytes.len() / (3 * std::mem::size_of::<f32>()), 3), bytemuck::cast_slice(bytes)).unwrap()
}

static _MAGMA_BYTES: &'static [u8] = include_bytes_aligned!(8, "color_data/magma.raw");
static MAGMA: OnceLock<ArrayView2<'static, f32>> = OnceLock::new();

pub fn magma() -> ArrayView2<'static, f32> {
    *MAGMA.get_or_init(|| color_data_as_arr(_MAGMA_BYTES))
}

#[inline]
pub fn normed_float_to_u8<F: Float + ToPrimitive>(val: F) -> u8 {
    let scale = F::from(256).unwrap();
    (val.clamp(F::zero(), F::one() - F::epsilon()) * scale).floor().to_u8().expect("Invalid value")
}

pub fn apply_cmap_float<'a, 'b, F: Float + ToPrimitive>(cmap: ArrayView2<'a, f32>, arr: ArrayViewD<'b, F>) -> ArrayD<f32> {
    assert!(cmap.shape() == &[256, 3]);
    let mut out_shape = arr.shape().to_vec(); out_shape.push(4);
    let mut out: ArrayD<f32> = ArrayD::zeros(out_shape);

    // TODO handle nans here

    Zip::from(arr)
        .and(out.rows_mut())
        .for_each(|v, mut out| {
            let i = normed_float_to_u8(*v) as usize;
            out.view_mut().slice_mut(s![..3]).assign(&cmap.index_axis(Axis(0), i));
            out.view_mut()[3] = 1.;
         });

    out
}

pub fn apply_cmap_u8<'a, 'b, F: Float + ToPrimitive>(cmap: ArrayView2<'a, f32>, arr: ArrayViewD<'b, F>) -> ArrayD<u8> {
    let out = apply_cmap_float(cmap, arr);
    out.mapv(normed_float_to_u8)
}

/*
    hsv_to_rgb:

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
*/

