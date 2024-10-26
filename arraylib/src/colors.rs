
use std::collections::HashMap;
use std::sync::OnceLock;

use num::{Float, ToPrimitive};
use ndarray::{ArrayViewD, ArrayView1, ArrayView2, ArrayViewMut1, ArrayD, Zip, Axis, s};
use include_bytes_aligned::include_bytes_aligned;
use paste::paste;

static NAMED_COLORS: OnceLock<HashMap<&'static str, ArrayView1<'static, f32>>> = OnceLock::new();

pub fn named_colors() -> &'static HashMap<&'static str, ArrayView1<'static, f32>> {
    NAMED_COLORS.get_or_init(|| {
        [
            ("empty",         ArrayView1::from(&[0.0, 0.0, 0.0, 0.0f32])),
            ("black",         ArrayView1::from(&[0.0, 0.0, 0.0, 1.0f32])),
            ("white",         ArrayView1::from(&[1.0, 1.0, 1.0, 1.0f32])),
            ("gray",          ArrayView1::from(&[0.666, 0.666, 0.666, 1.0f32])),
            ("red",           ArrayView1::from(&[1.0, 0.0, 0.0, 1.0f32])),
            ("green",         ArrayView1::from(&[0.0, 1.0, 0.0, 1.0f32])),
            ("blue",          ArrayView1::from(&[0.0, 0.0, 1.0, 1.0f32])),

            ("tab:blue",      ArrayView1::from(&[0.12156863, 0.46666667, 0.70588235, 1.0f32])),
            ("tab:orange",    ArrayView1::from(&[1.00000000, 0.49803922, 0.05490196, 1.0f32])),
            ("tab:green",     ArrayView1::from(&[0.17254902, 0.62745098, 0.17254902, 1.0f32])),
            ("tab:red",       ArrayView1::from(&[0.83921569, 0.15294118, 0.15686275, 1.0f32])),
            ("tab:purple",    ArrayView1::from(&[0.58039216, 0.40392157, 0.74117647, 1.0f32])),
            ("tab:brown",     ArrayView1::from(&[0.54901961, 0.33725490, 0.29411765, 1.0f32])),
            ("tab:pink",      ArrayView1::from(&[0.89019608, 0.46666667, 0.76078431, 1.0f32])),
            ("tab:gray",      ArrayView1::from(&[0.49803922, 0.49803922, 0.49803922, 1.0f32])),
            ("tab:olive",     ArrayView1::from(&[0.73725490, 0.74117647, 0.13333333, 1.0f32])),
            ("tab:cyan",      ArrayView1::from(&[0.09019608, 0.74509804, 0.81176471, 1.0f32])),

            ("sasha:red",     ArrayView1::from(&[0.90196078, 0.09803922, 0.29411765, 1.0f32])),
            ("sasha:orange",  ArrayView1::from(&[0.96078431, 0.50980392, 0.19215686, 1.0f32])),
            ("sasha:yellow",  ArrayView1::from(&[1.00000000, 0.88235294, 0.09803922, 1.0f32])),
            ("sasha:lime",    ArrayView1::from(&[0.74901961, 0.93725490, 0.27058824, 1.0f32])),
            ("sasha:green",   ArrayView1::from(&[0.23529412, 0.70588235, 0.29411765, 1.0f32])),
            ("sasha:cyan",    ArrayView1::from(&[0.25882353, 0.83137255, 0.95686275, 1.0f32])),
            ("sasha:blue",    ArrayView1::from(&[0.26274510, 0.38823529, 0.84705882, 1.0f32])),
            ("sasha:purple",  ArrayView1::from(&[0.56862745, 0.11764706, 0.70588235, 1.0f32])),
            ("sasha:magenta", ArrayView1::from(&[0.94117647, 0.19607843, 0.90196078, 1.0f32])),
            ("sasha:grey",    ArrayView1::from(&[0.66274510, 0.66274510, 0.66274510, 1.0f32])),
            ("sasha:navy",    ArrayView1::from(&[0.00000000, 0.00000000, 0.45882353, 1.0f32])),
        ].into_iter().collect()
    })
}

fn color_data_as_arr(bytes: &'static [u8]) -> ArrayView2<'static, f32> {
    ArrayView2::from_shape((bytes.len() / (3 * std::mem::size_of::<f32>()), 3), bytemuck::cast_slice(bytes)).unwrap()
}

macro_rules! impl_colormaps {
    ( $name:ident ) => { paste! {
        static [<$name:upper _BYTES>]: &'static [u8] = include_bytes_aligned!(8, concat!("color_data/", stringify!($name), ".raw"));
        static [<$name:upper>]: OnceLock<ArrayView2<'static, f32>> = OnceLock::new();

        pub fn $name() -> ArrayView2<'static, f32> {
            *[<$name:upper>].get_or_init(|| color_data_as_arr([<$name:upper _BYTES>]))
        }
    } };

    ( $( $name:ident ),+ ) => {
        $( impl_colormaps!($name); )+

        pub fn get_cmap(name: &str) -> Result<ArrayView2<'static, f32>, String> {
            Ok(match name {
                $( stringify!($name) => $name(), )+
                _ => return Err(format!("Unknown colormap '{}'", name)),
            })
        }
    }
}

impl_colormaps!(cividis, inferno, magma, plasma, viridis, sinebow, hue);

#[inline]
pub fn normed_float_to_u8<F: Float + ToPrimitive>(val: F) -> u8 {
    let scale = F::from(256).unwrap();
    (val.clamp(F::zero(), F::one() - F::epsilon()) * scale).floor().to_u8().expect("Invalid value")
}

#[inline]
fn apply_cmap_inner<F: Float + ToPrimitive>(
    val: F,
    mut out: ArrayViewMut1<'_, f32>,
    cmap: &ArrayView2<'_, f32>,
    min_color: Option<ArrayView1<'_, f32>>,
    max_color: Option<ArrayView1<'_, f32>>,
    invalid_color: ArrayView1<'_, f32>,
) {
    if let Some(min_c) = min_color {
        if val < F::zero() {
            out.assign(&min_c);
            return;
        }
    }
    if let Some(max_c) = max_color {
        if val > F::one() {
            out.assign(&max_c);
            return;
        }
    }
    if val.is_nan() {
        out.assign(&invalid_color);
        return;
    }
    let i = normed_float_to_u8(val) as usize;
    out.slice_mut(s![..3]).assign(&cmap.index_axis(Axis(0), i));
    out[3] = 1.;
}

pub fn apply_cmap_float<F: Float + ToPrimitive>(
    cmap: ArrayView2<'_, f32>, arr: ArrayViewD<'_, F>,
    min_color: Option<ArrayView1<'_, f32>>,
    max_color: Option<ArrayView1<'_, f32>>,
    invalid_color: ArrayView1<'_, f32>,
) -> ArrayD<f32> {
    assert!(cmap.shape() == &[256, 3]);
    let mut out_shape = arr.shape().to_vec(); out_shape.push(4);
    let mut out: ArrayD<f32> = ArrayD::zeros(out_shape);

    Zip::from(arr)
        .and(out.rows_mut())
        .for_each(|v, mut out| {
            apply_cmap_inner(*v, out.view_mut(), &cmap, min_color, max_color, invalid_color);
         });

    out
}

pub fn apply_cmap_u8<F: Float + ToPrimitive>(
    cmap: ArrayView2<'_, f32>, arr: ArrayViewD<'_, F>,
    min_color: Option<ArrayView1<'_, f32>>,
    max_color: Option<ArrayView1<'_, f32>>,
    invalid_color: ArrayView1<'_, f32>,
) -> ArrayD<u8> {
    let out = apply_cmap_float(cmap, arr, min_color, max_color, invalid_color);
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

