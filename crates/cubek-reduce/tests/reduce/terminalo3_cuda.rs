//! CUDA parity regressions extracted from TerminalO3 autotune artifacts.

use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    cuda::CudaRuntime,
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, shape},
};
use cubek_reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    reduce,
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
    shared_sum,
};
use half::f16;

fn f16_dtypes() -> ReduceDtypes {
    ReduceDtypes {
        input: f16::as_type_native_unchecked().storage_type(),
        output: f16::as_type_native_unchecked().storage_type(),
        accumulation: f32::as_type_native_unchecked().storage_type(),
    }
}

fn filled_f16_tensor(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    value: f32,
) -> TensorHandle<CudaRuntime> {
    let num_elements = shape.iter().product();
    let values = vec![f16::from_f32(value); num_elements];
    let layout = client.create_tensor_from_slice(f16::as_bytes(&values), shape.clone(), 2);
    TensorHandle::new(
        layout.memory,
        shape,
        layout.strides,
        f16::as_type_native_unchecked(),
    )
}

fn filled_f32_tensor(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    value: f32,
) -> TensorHandle<CudaRuntime> {
    let num_elements = shape.iter().product();
    let values = vec![value; num_elements];
    let layout = client.create_tensor_from_slice(f32::as_bytes(&values), shape.clone(), 4);
    TensorHandle::new(
        layout.memory,
        shape,
        layout.strides,
        f32::as_type_native_unchecked(),
    )
}

fn read_f16(client: &ComputeClient<CudaRuntime>, output: TensorHandle<CudaRuntime>) -> Vec<f32> {
    let bytes = client.read_one_unchecked_tensor(output.into_copy_descriptor());
    f16::from_bytes(&bytes)
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

fn read_f32(client: &ComputeClient<CudaRuntime>, output: TensorHandle<CudaRuntime>) -> Vec<f32> {
    let bytes = client.read_one_unchecked_tensor(output.into_copy_descriptor());
    f32::from_bytes(&bytes).to_vec()
}

fn output_shape(input_shape: &Shape, axis: usize) -> Shape {
    let mut shape = input_shape.clone();
    shape[axis] = 1;
    shape
}

fn unit_strategy(vectorized_output: bool) -> ReduceStrategy {
    ReduceStrategy {
        routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: vectorized_output,
        },
    }
}

fn plane_strategy(vectorized_output: bool) -> ReduceStrategy {
    ReduceStrategy {
        routine: RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
            independent: true,
        })),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: vectorized_output,
        },
    }
}

fn cube_strategy(vectorized_output: bool) -> ReduceStrategy {
    ReduceStrategy {
        routine: RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
            use_planes: true,
        })),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: vectorized_output,
        },
    }
}

fn assert_reduce_dim_sum(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    axis: usize,
    strategy: ReduceStrategy,
) {
    let input = filled_f16_tensor(client, shape.clone(), 1.0);
    let output = filled_f16_tensor(client, output_shape(&shape, axis), 0.0);

    reduce::<CudaRuntime>(
        client,
        input.binding(),
        output.clone().binding(),
        axis,
        strategy,
        ReduceOperationConfig::Sum,
        f16_dtypes(),
    )
    .unwrap();

    let expected = shape[axis] as f32;
    let actual = read_f16(client, output);
    let mismatches = actual
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != expected)
        .take(16)
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "Reduce-dim parity mismatches at {mismatches:?}; expected {expected}",
    );
}

#[test]
fn test_terminalo3_sum_one_shot_parity() {
    let client = CudaRuntime::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().elem_type();

    for (length, cube_count) in [
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 8),
        (16, 16),
        (16, 32),
        (16, 64),
    ] {
        let input = filled_f32_tensor(&client, shape![length], 1.0);
        let output = filled_f32_tensor(&client, shape![1], 0.0);

        shared_sum(
            &client,
            input.binding(),
            output.clone().binding(),
            cube_count,
            dtype,
        )
        .unwrap();

        assert_eq!(
            vec![length as f32],
            read_f32(&client, output),
            "One-shot parity failed for length {length} and cube count {cube_count}",
        );
    }
}

#[test]
fn test_terminalo3_sum_chained_parity() {
    let client = CudaRuntime::client(&Default::default());
    let mut input = filled_f16_tensor(&client, shape![8, 16, 4], 1.0);

    for axis in [2, 0, 1] {
        let output = filled_f16_tensor(&client, output_shape(input.shape(), axis), 0.0);
        reduce::<CudaRuntime>(
            &client,
            input.binding(),
            output.clone().binding(),
            axis,
            unit_strategy(false),
            ReduceOperationConfig::Sum,
            f16_dtypes(),
        )
        .unwrap();
        input = output;
    }

    assert_eq!(vec![512.0], read_f16(&client, input));
}

#[test]
fn test_terminalo3_reduce_dim_routine_parity() {
    let client = CudaRuntime::client(&Default::default());

    for strategy in [
        unit_strategy(false),
        plane_strategy(false),
        cube_strategy(false),
    ] {
        assert_reduce_dim_sum(&client, shape![512, 1024], 1, strategy.clone());
        assert_reduce_dim_sum(&client, shape![1024, 512], 0, strategy);
    }
}

#[test]
fn test_terminalo3_reduce_dim_vectorized_output_parity() {
    let client = CudaRuntime::client(&Default::default());

    for strategy in [
        unit_strategy(true),
        plane_strategy(true),
        cube_strategy(true),
    ] {
        assert_reduce_dim_sum(&client, shape![512, 1024], 1, strategy);
    }
}
