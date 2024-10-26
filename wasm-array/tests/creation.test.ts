import {describe, expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("array", () => {
    let arr = np.array([true, 2, 3, 4, 5]);
    let arrcopy = np.array(arr);

    expect(arr.toString()).toBe("Array int64 [1, 2, 3, 4, 5]");
    expect(arrcopy.toString()).toBe("Array int64 [1, 2, 3, 4, 5]");

    arr = np.array([1, 2, 3.5, 4, 5]);
    expect(arr.toString()).toBe("Array float64 [1, 2, 3.5, 4, 5]");

    arr = np.array([[1, 2], [4, 5]], 'complex64');
    expect(arr.toString()).toBe(
`Array complex64
[[1+0i, 2+0i],
 [4+0i, 5+0i]]`
    );

    arr = np.array([true, true, false, true, true]);
    expect(arr.toString()).toBe("Array bool [true, true, false, true, true]");
})

test("array_bignum", () => {
    let arr = np.array([1n, 2n, 3n, 4n, 5n]);
    expect(arr.toString()).toBe("Array int64 [1, 2, 3, 4, 5]");

    let val = 99999999999999999999999999n;

    expect(() => {
        np.array([1n, 2n, 3, 99999999999999999999999999n, 50000000n]);
    }).toThrowError("BigInt '99999999999999999999999999' overflows i64");
})

test("zeros", () => {
    let arr = np.zeros([5], 'complex64');

    expect(arr.toString()).toBe("Array complex64 [0+0i, 0+0i, 0+0i, 0+0i, 0+0i]");
})

test("ones", () => {
    let arr = np.ones([2, 3, 1], 'bool');

    expect(arr.toString()).toBe(
`Array bool
[[[true],
  [true],
  [true]],

 [[true],
  [true],
  [true]]]`
    );
})

test("indices", () => {
    let [arr1, arr2] = np.indices([2, 2]);

    expect(arr1.toString()).toBe(
`Array int64
[[0, 0],
 [1, 1]]`
    );
    expect(arr1).arrayEqual([[0, 0], [1, 1]]);

    expect(arr2.toString()).toBe(
`Array int64
[[0, 1],
 [0, 1]]`
    );
    expect(arr2).arrayEqual([[0, 1], [0, 1]]);
})

test("arange", () => {
    let arr = np.arange(5, undefined, 'int32');
    expect(arr.toString()).toBe("Array int32 [0, 1, 2, 3, 4]");

    arr = np.arange(1, 6);
    expect(arr.toString()).toBe("Array int64 [1, 2, 3, 4, 5]");
    expect(arr).arrayEqual([1, 2, 3, 4, 5]);
})

test("linspace", () => {
    let arr = np.linspace(0., 10., 11, 'float32');
    expect(arr).arrayAlmostEqual(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'float32'));
})

test("logspace", () => {
    let arr = np.logspace(0., 2., 10, 'float32');
    expect(arr).arrayAlmostEqual(np.array([
        1.0, 1.6681006, 2.7825594, 4.641589, 7.742637, 12.915499, 21.54435, 35.93814, 59.948425, 100.0
    ], 'float32'), 1e-6);
})

test("geomspace", () => {
    let arr = np.geomspace(1.0, 100.0, 10, 'float32');
    expect(arr).arrayAlmostEqual(np.array([
        1.0, 1.6681006, 2.7825594, 4.641589, 7.742637, 12.915499, 21.54435, 35.93814, 59.948425, 100.0
    ], 'float32'), 1e-6);
})

test("eye", () => {
    let arr = np.eye(3);

    expect(arr.toString()).toBe(
`Array float64
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]`
    );
})

describe("from_interchange", () => {
    let buf = new ArrayBuffer(10);
    let arr = new Uint16Array(buf);
    arr.set([1, 2, 3, 4, 5]);

    const dataFormats = [
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
        "AQACAAMABAAFAA==",
        buf,
        arr,
    ];

    for (const data of dataFormats) {
        test(`data=${data} ${data instanceof ArrayBuffer}`, () => {
            let arr = np.from_interchange({
                data: data,
                strides: null,
                typestr: '<u2',
                shape: [5],
                version: 3
            });

            expect(arr).arrayEqual(np.array([1, 2, 3, 4, 5], 'uint16'));
        })
    }
})

test("toNestedArray", () => {
    let arr = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]);
    expect(arr.toNestedArray()).toEqual([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]);

    arr = np.array([true, true, false, true]);
    expect(arr.toNestedArray()).toEqual([true, true, false, true]);

    arr = np.array(5.6);
    expect(arr.toNestedArray()).toEqual(5.6);
})