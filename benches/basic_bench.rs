//! Basic benchmark suite

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use core::tensor::{Tensor, TensorType, Shape, TensorData};
use core::ops::{Add, MatMul};

fn bench_tensor_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_add");

    for size in [64, 128, 256, 512, 1024].iter() {
        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(*size, *size),
            TensorData::F32(vec![1.0; size * size]),
        ).unwrap();

        let b = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(*size, *size),
            TensorData::F32(vec![2.0; size * size]),
        ).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, _| {
            bencher.iter(|| {
                let op = Add;
                op.apply(black_box(&a), black_box(&b)).unwrap()
            })
        });
    }

    group.finish();
}

fn bench_tensor_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_matmul");

    for size in [32, 64, 128, 256].iter() {
        let a = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(*size, *size),
            TensorData::F32(vec![1.0; size * size]),
        ).unwrap();

        let b = Tensor::new(
            None,
            TensorType::F32,
            Shape::matrix(*size, *size),
            TensorData::F32(vec![2.0; size * size]),
        ).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, _| {
            bencher.iter(|| {
                let op = MatMul;
                op.apply(black_box(&a), black_box(&b)).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tensor_add, bench_tensor_matmul);
criterion_main!(benches);
