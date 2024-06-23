import { to_dtype, ones, test, expr } from './pkg';

//console.log(test());
//console.log(to_dtype("uint8").toString());

//console.log(ones([1, 5], "float32").toString());

//console.log(exec(["1. + ", " + 5"], [ones([1, 5], "float32")]).toString());

let a = ones([1, 5], "int32");
console.log(expr`1. + 5 * ${a}`.toString());