import { ones, zeros, expr, linspace, from_interchange } from './pkg';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext("2d");

let arr = linspace(-1.5, 1.5, 500, "float32").broadcast_to([100, 500]);
arr = expr`abs(${arr}) * 2/3`;

const imageData = ctx.createImageData(500, 100);
imageData.data.set(arr.apply_cmap());
ctx.putImageData(imageData, 0, 0);