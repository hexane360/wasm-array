

pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        if (-axis as usize) > ndim { panic!("Axis {} out of range for ndim {}", axis, ndim); }
        ndim - (-axis as usize)
    } else {
        if axis as usize + 1 > ndim { panic!("Axis {} out of range for ndim {}", axis, ndim); }
        axis as usize
    }
}