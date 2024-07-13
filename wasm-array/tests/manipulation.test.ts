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