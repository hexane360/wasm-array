import {describe, expect, test} from '@jest/globals';

import * as np from '../pkg/wasm_array';

test("boolean exprs", () => {
    expect(np.expr`${1} | ${0} & ${0}`).arrayEqual(1);
    expect(np.expr`(${true} | ${false}) & ${false}`).arrayEqual(false);
    expect(np.expr`${10} >= ${5} & ${9} >= ${1}`).arrayEqual(true);
})