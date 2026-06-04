//! CUDA parity regressions extracted from TerminalO3 autotune artifacts.

use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    cuda::CudaRuntime,
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, shape},
};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    launch::{Strategy, launch_ref},
};
use cubek_std::InputBinding;
use half::f16;

const LARGE_M: usize = 2_097_153;

fn f16_dtypes() -> MatmulElems {
    let f16 = f16::as_type_native_unchecked().storage_type();
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f16,
        rhs: f16,
        out: f16,
    })
}

fn filled_tensor(
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

fn read_f16(client: &ComputeClient<CudaRuntime>, output: TensorHandle<CudaRuntime>) -> Vec<f32> {
    let bytes = client.read_one_unchecked_tensor(output.into_copy_descriptor());
    f16::from_bytes(&bytes)
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

#[test]
fn test_terminalo3_naive_large_m_axis_parity() {
    let client = CudaRuntime::client(&Default::default());
    let lhs = filled_tensor(&client, shape![LARGE_M, 1], 1.0);
    let rhs = filled_tensor(&client, shape![1, 1], 1.0);
    let out = filled_tensor(&client, shape![LARGE_M, 1], 0.0);
    let dtype = f16::as_type_native_unchecked().storage_type();

    launch_ref(
        &Strategy::Naive,
        &client,
        InputBinding::Normal(lhs.binding(), dtype),
        InputBinding::Normal(rhs.binding(), dtype),
        out.clone().binding(),
        &mut f16_dtypes(),
    )
    .unwrap();

    let actual = read_f16(&client, out);
    let mismatches = actual
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 1.0)
        .take(16)
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "Naive large-axis parity mismatches at {mismatches:?}",
    );
}
