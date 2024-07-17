import {expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("ceil", () => {
    expect(np.ceil([[1.5, 2.3, 1.1, -1.5]])).arrayEqual(np.array([[2.0, 3.0, 2.0, -1.0]], 'float64'))
})

test("minimum", () => {
    expect(np.minimum([1, 5, 6], [4, 2, 3])).arrayEqual(np.array([1, 2, 3], 'int64'))

    expect(np.expr`minimum(${np.array([1, 5, 6], 'float64')}, ${[4, 2, 3]})`).arrayEqual(np.array([1, 2, 3], 'float64'))
})

test("maximum", () => {
    expect(np.maximum([1, 5, 6], [4, 2, 3])).arrayEqual(np.array([4, 5, 6], 'int64'))

    expect(np.expr`maximum(${np.array([1, 5, 6], 'float64')}, ${[4, 2, 3]})`).arrayEqual(np.array([4, 5, 6], 'float64'))
})