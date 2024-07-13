import {describe, expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

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

    expect(arr2.toString()).toBe(
`Array int64
[[0, 1],
 [0, 1]]`
    );
})

test("arange", () => {
    let arr = np.arange(5, undefined, 'int32');
    expect(arr.toString()).toBe("Array int32 [0, 1, 2, 3, 4]");

    arr = np.arange(1, 6);
    expect(arr.toString()).toBe("Array int64 [1, 2, 3, 4, 5]");
})

test("linspace", () => {
    let arr = np.linspace(0., 10., 11, 'float32');
    expect(arr.toString()).toBe("Array float32 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
})

// TODO: logspace, geomspace (need array almost equals)

test("eye", () => {
    let arr = np.eye(3);

    expect(arr.toString()).toBe(
`Array float64
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]`
    );
})