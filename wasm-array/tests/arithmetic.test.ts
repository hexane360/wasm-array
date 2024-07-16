import {expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("ceil", () => {
    expect(np.ceil([[1.5, 2.3, 1.1, -1.5]])).arrayEqual(np.array([[2.0, 3.0, 2.0, -1.0]], 'float64'))
})