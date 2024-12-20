import {expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("ceil", () => {
    expect(np.ceil([[1.5, 2.3, 1.1, -1.5]])).arrayEqual(np.array([[2.0, 3.0, 2.0, -1.0]], 'float64'))
})

test("minimum", () => {
    expect(np.minimum([1, 5, 6], [4, 2, 3])).arrayEqual(np.array([1, 2, 3], 'int64'))

    expect(np.expr`minimum(${np.array([1, 5, 6], 'float64')}, ${[4, 2, 3]})`).arrayEqual(np.array([1, 2, 3], 'float64'))

    expect(np.minimum([np.nan(), 2, 3], [2, np.nan(), 4])).arrayAlmostEqual(np.array([np.nan(), np.nan(), 3], 'float64'));
    expect(np.nanminimum([np.nan(), 2, 3], [2, np.nan(), 4])).arrayAlmostEqual(np.array([2, 2, 3], 'float64'));
})

test("maximum", () => {
    expect(np.maximum([1, 5, 6], [4, 2, 3])).arrayEqual(np.array([4, 5, 6], 'int64'))

    expect(np.expr`maximum(${np.array([1, 5, 6], 'float64')}, ${[4, 2, 3]})`).arrayEqual(np.array([4, 5, 6], 'float64'))

    expect(np.maximum([np.nan(), 2, 3], [2, np.nan(), 4])).arrayAlmostEqual(np.array([np.nan(), np.nan(), 4], 'float64'));
    expect(np.nanmaximum([np.nan(), 2, 3], [2, np.nan(), 4])).arrayAlmostEqual(np.array([2, 2, 4], 'float64'));
})

test("exp", () => {
    expect(np.exp_(np.array([1, 2, -1, -5], 'float32'))).arrayAlmostEqual(
        np.array([2.7182817, 7.389056, 0.36787945, 0.006737947], 'float32'),
    1e-6, 1e-8);

    let arr = np.array([0.0, 1.0, np.expr`pi*-1j`, np.expr`pi*1j`], 'complex64');
    let expected = np.array([1.0, 2.7182817, -1.0, -1.0], 'complex64');

    expect(np.exp_(arr)).arrayAlmostEqual(expected, 1e-8, 1e-7);
    expect(np.expr`exp(${arr})`).arrayAlmostEqual(expected, 1e-8, 1e-7);
})

test("log", () => {
    expect(np.log_(np.array([1, 2, 3, 4, 5], 'float32'))).arrayAlmostEqual(
        np.array([0.        , 0.69314718, 1.09861229, 1.38629436, 1.60943791], 'float32')
    );

    expect(np.log2_([1, 2, 3, 4, 5])).arrayAlmostEqual(
        np.array([0.        , 1.        , 1.5849625 , 2.        , 2.32192809], 'float64')
    );

    expect(np.log10_(np.array([2, 4, 6, 8, 10], 'complex64'))).arrayAlmostEqual(
        np.array([0.30102998, 0.60205999, 0.77815125, 0.90308999, 1.        ], 'complex64')
    );

    let arr = np.array([1, 2, 3, 4, 5], 'float32')
    let expected = np.array([0.        , 0.69314718, 1.09861229, 1.38629436, 1.60943791], 'float32')
    expect(np.expr`log(${arr})`).arrayAlmostEqual(expected);
})

test("trig", () => {
    let arr = np.array([0., np.expr`pi/2`, np.expr`2*pi`, 1.], 'float64');

    let expected = np.array([0., 1., 0., 0.8414709848078965], 'float64');
    expect(np.expr`sin(${arr})`).arrayAlmostEqual(expected, 1e-10, 1e-10);
    expect(np.sin_(arr)).arrayAlmostEqual(expected, 1e-10, 1e-10);

    expected = np.array([1., 0., 1., 0.54030230586], 'float64');
    expect(np.expr`cos(${arr})`).arrayAlmostEqual(expected, 1e-10, 1e-10);
    expect(np.cos_(arr)).arrayAlmostEqual(expected, 1e-10, 1e-10);
})

test("interp", () => {
    expect(np.interp([[1., 2.], [3.5, 4.5], [-1, 10]], [0., 2., 3., 5., 8.], [0., 5., 2., 8., 15.]))
        .arrayAlmostEqual([[2.5, 5.], [3.5, 6.5], [0., 15.]]);

    // test codepath without pre-computed slopes, with manually-specified endpoints
    expect(np.interp([-1, 3.5, 10, np.nan()], [0., 1., 2., 3., 5., 8.], [0., 2.5, 5., 2., 8., 15.], 10, -10))
        .arrayAlmostEqual([10, 3.5, -10, np.nan()]);
})

test("interpn", () => {
    expect(np.interpn([[0., 2., 3., 5., 8.]], [0., 5., 2., 8., 15.], [[[1.], [2.]], [[3.5], [4.5]], [[-1], [10]]], -1))
        .arrayAlmostEqual([[2.5, 5.], [3.5, 6.5], [-1., -1.]]);

    const xs = np.array([0.0, 2.0, 3.0, 6.0, 10.0]);
    const ys = np.array([0.0, 1.0, 5.0, 6.0, 8.0]);
    const values = np.array([
        [  3.,   9.,  -8.,  -9.,  29.],
        [ -1.,   2.,  13., -16.,  20.],
        [  9.,   6.,  10.,   1.,  -0.],
        [ -0.,  16.,   6., -11.,  -5.],
        [  3.,  -2.,  19.,  17.,  18.],
    ]);

    expect(np.interpn([xs, ys], values, [10., 8.]))
        .arrayAlmostEqual(np.array(18., 'float64'))

    expect(np.interpn([xs, ys], values, [[0.5, 0.5], [10., 8.], [5., 6.], [-1., 5.], [5., -1.], [7.5, 7.]], -5))
        .arrayAlmostEqual([4.625, 18., -7., -5., -5., 1.5625])
})