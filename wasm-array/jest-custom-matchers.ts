import { expect } from '@jest/globals';
import type {MatcherFunction} from 'expect';

import * as np from './pkg/wasm_array';

function _shapeEqual(lhs: Uint32Array, rhs: Uint32Array): boolean {
    if (lhs.length != rhs.length) { return false; }
    for (let i = 0; i < lhs.length; i++) {
        if (lhs[i] !== rhs[i]) {
            return false;
        }
    }
    return true;
} 

const arrayEqual: MatcherFunction<[np.ArrayLike]> = function (received: unknown, expected: np.ArrayLike) {
    const actual = received as np.NArray;
    let expected_arr = np.array(expected);

    let fail = false;
    let msg: string | null = null;

    if (!_shapeEqual(actual.shape, expected_arr.shape)) {
        fail = true;
        msg = "Shape mismatch"
    } else if (actual.dtype.toString() !== expected_arr.dtype.toString()) {
        fail = true;
        msg = "dtype mismatch"
    } else if (!np.allequal(actual, expected_arr)) {
        fail = true;
    }
    if (fail) {
        msg = (msg === null) ? "" : ` (${msg})`;
        return {
            pass: false,
            message: () => `Arrays not equal${msg}
Received: ${actual.toString()}

Expected: ${expected_arr.toString()}`,
        }
    }
    return {
        pass: true,
        message: () => "Expected arrays to be equal",
    };
}

const arrayAlmostEqual: MatcherFunction<[np.ArrayLike, number, number]> = function (received, expected: np.ArrayLike, rtol: number = 1e-8, atol: number = 0.0) {
    const actual = received as np.NArray;
    let expected_arr = np.array(expected);

    let fail = false;
    let msg: string | null = null;

    if (!_shapeEqual(actual.shape, expected_arr.shape)) {
        fail = true;
        msg = "Shape mismatch"
    } else if (actual.dtype.toString() != expected_arr.dtype.toString()) {
        fail = true;
        msg = "dtype mismatch"
    } else if (!np.allclose(actual, expected_arr, rtol, atol)) {
        fail = true;
    }
    if (fail) {
        msg = (msg === null) ? "" : ` (${msg})`;
        return {
            pass: false,
            message: () => `Arrays not almost equal${msg}
Received: ${actual.toString()}

Expected: ${expected_arr.toString()}`,
        }
    }
    return {
        pass: true,
        message: () => "Expected arrays to be almost equal",
    };
}

expect.extend({
    arrayEqual, arrayAlmostEqual
})

declare module 'expect' {
    interface AsymmetricMatchers {
        arrayEqual(expected: np.ArrayLike): void;
        arrayAlmostEqual(expected: np.ArrayLike, rtoi?: number, atoi?: number): void;
    }
    interface Matchers<R> {
        arrayEqual(expected: np.ArrayLike): R;
        arrayAlmostEqual(expected: np.ArrayLike, rtoi?: number, atoi?: number): R;
    }
}