import { ones, zeros, expr, linspace } from './pkg';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext("2d");

let arr = linspace(1., 10., 500, "float32").broadcast_to([100, 500]);
arr = expr`1 - ${arr}^-0.4`;

const imageData = ctx.createImageData(500, 100);
imageData.data.set(arr.apply_cmap());
ctx.putImageData(imageData, 0, 0);

