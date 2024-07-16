import {describe, expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("sum", () => {
    np.sum(np.array([10, -20, 40, 5, 2], 'int64'))
})