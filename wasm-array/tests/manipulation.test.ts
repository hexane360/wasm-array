import {describe, expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("ravel/flatten", () => {
    let arr = np.eye(4, 'int16');

    for (const result of [np.ravel(arr), arr.ravel(), arr.flatten()]) {
        expect(result.toString()).toBe("Array int16 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]");
    }
})

test("reshape", () => {
    let arr = np.arange(1, 7, 'int16');

    expect(np.reshape(arr, [2, 3]).toString()).toBe(
`Array int16
[[1, 2, 3],
 [4, 5, 6]]`
    );

    expect(np.reshape(arr, [2, -1]).toString()).toBe(
`Array int16
[[1, 2, 3],
 [4, 5, 6]]`
    );

    expect(() => {
        np.reshape(arr, [2, 5]);
    }).toThrowError("Cannot reshape array of size 6 into shape [2, 5]");

    expect(() => {
        np.reshape(arr, [1000, 1000, 1000, 1000, 1000]);
    }).toThrowError("Overflow evaluating shape [1000, 1000, 1000, 1000, 1000]");

    expect(() => {
        np.reshape(arr, [5, -1]);
    }).toThrowError("Cannot reshape array of size 6 into shape [5, -1]");
})

test("meshgrid", () => {
    const yvec = np.array([2, 8], 'uint64');
    const xvec = np.array([1, 2, 5, 10], 'int16');

    const [ymat, xmat] = np.meshgrid(yvec, xvec);

    expect(ymat).arrayEqual(np.array([[2, 2, 2, 2], [8, 8, 8, 8]], 'uint64'));
    expect(xmat).arrayEqual(np.array([[1, 2, 5, 10], [1, 2, 5, 10]], 'int16'));

    let [arr] = np.meshgrid([1, 2, 3, 4]);
    expect(arr).arrayEqual(np.array([1, 2, 3, 4], 'int64'));

    expect(() => {
        np.meshgrid([[1, 2], [2, 4]], [2, 4, 8])
    }).toThrowError("'meshgrid' requires 1D input arrays");
})

test("stack", () => {
    const x = np.array([[1, 2], [3, 4]]);
    const y = np.array([[5, 6], [7, 8]]);

    expect(np.stack([x, y], -1)).arrayEqual([
        [[1, 5], [2, 6]],
        [[3, 7], [4, 8]],
    ]);

    expect(np.stack([x, y])).arrayEqual([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]);
})

test("broadcast_shapes", () => {
    expect(np.broadcast_shapes()).toEqual([]);
    expect(np.broadcast_shapes([], [], [])).toEqual([]);
    expect(np.broadcast_shapes([1, 2, 3], [5, 1, 1], [])).toEqual([5, 2, 3]);

    expect(() => np.broadcast_shapes([1, 2, 3], [1, 1, 4])).toThrowError("Unable to broadcast shapes [1, 2, 3] and [1, 1, 4]");
    expect(() => np.broadcast_shapes([1, 2, 3], [1], [1, 1, 4])).toThrowError("Unable to broadcast shapes [1, 2, 3], [1], and [1, 1, 4]");
})

test("broadcast_to", () => {
    expect(np.broadcast_to([1, 2, 3, 4], [1, 1, 4])).arrayEqual([[[1, 2, 3, 4]]]);
})

test("broadcast_arrays", () => {
    const [arr1, arr2] = np.broadcast_arrays([[1, 2, 3, 4]], [[4], [3], [2], [1]]);

    expect(arr1).arrayEqual([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]);
    expect(arr2).arrayEqual([[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1, 1]]);
})