import { to_dtype, ones, zeros, test, expr, from_interchange } from './pkg';

zeros([1, 5], "float32");
zeros([1, 5], "u8");

//console.log(exec(["1. + ", " + 5"], [ones([1, 5], "float32")]).toString());

const buffer = new ArrayBuffer(8);

let view = new Uint8Array(new ArrayBuffer(16));
view = view.map((_, index) => index * (1 - index % 2))

console.log(view.version);

let a = from_interchange({
    'data': view,
    'typestr': '<u2',
    'shape': [1, 8],
    'strides': [8, -1],
});
console.log(a.toString());
console.log(a.toJSON());
console.log(a.toInterchange());
//console.log(expr`${a}`.toString());