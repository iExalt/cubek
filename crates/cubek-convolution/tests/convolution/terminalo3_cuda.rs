//! CUDA parity regressions extracted from TerminalO3 autotune artifacts.

use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    cuda::CudaRuntime,
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, shape},
};
use cubek_convolution::{
    AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref,
};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::InputBinding;
use half::f16;

use super::launcher_strategy::ConvolutionCase;

const REPETITIONS: usize = 2;
const TOLERANCE: f32 = 0.5;
const FORWARD_INPUT_VALUE: f32 = 0.25;
const FORWARD_WEIGHT_VALUE: f32 = 0.25;
const WGRAD_INPUT_VALUE: f32 = 0.25;
const WGRAD_OUT_GRAD_VALUE: f32 = 0.25;

fn f16_dtypes() -> MatmulElems {
    let f16 = f16::as_type_native_unchecked().storage_type();
    MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f16,
        rhs: f16,
        out: f16,
    })
}

fn rollout_case() -> ConvolutionCase {
    ConvolutionCase {
        batches: 4,
        in_h: 32,
        in_w: 32,
        in_channels: 256,
        out_channels: 64,
        kernel_size: [3, 3],
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        has_bias: true,
    }
}

fn learner_cmma_case() -> ConvolutionCase {
    ConvolutionCase {
        batches: 256,
        in_h: 32,
        in_w: 32,
        in_channels: 32,
        out_channels: 32,
        kernel_size: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        has_bias: true,
    }
}

fn compact_case() -> ConvolutionCase {
    ConvolutionCase {
        batches: 4,
        in_h: 32,
        in_w: 32,
        in_channels: 32,
        out_channels: 32,
        kernel_size: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        has_bias: true,
    }
}

fn args(case: &ConvolutionCase) -> ConvolutionArgs<2> {
    ConvolutionArgs {
        stride: case.stride,
        padding: case.padding,
        dilation: case.dilation,
    }
}

fn strategy(algorithm: ConvAlgorithm, tile_kind: AcceleratedTileKind) -> Strategy {
    Strategy::Inferred {
        algorithm,
        tile_kind,
    }
}

fn input_shape(case: &ConvolutionCase) -> Shape {
    shape![case.batches, case.in_h, case.in_w, case.in_channels]
}

fn weight_shape(case: &ConvolutionCase) -> Shape {
    shape![
        case.out_channels,
        case.kernel_size[0],
        case.kernel_size[1],
        case.in_channels
    ]
}

fn output_shape(case: &ConvolutionCase) -> Shape {
    shape![case.batches, case.out_h(), case.out_w(), case.out_channels]
}

fn ones_tensor(client: &ComputeClient<CudaRuntime>, shape: Shape) -> TensorHandle<CudaRuntime> {
    filled_tensor(client, shape, 1.0)
}

fn filled_tensor(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    value: f32,
) -> TensorHandle<CudaRuntime> {
    let num_elements = shape.iter().product();
    tensor_from_f32(client, shape, vec![value; num_elements])
}

fn zeros_tensor(client: &ComputeClient<CudaRuntime>, shape: Shape) -> TensorHandle<CudaRuntime> {
    let num_elements = shape.iter().product();
    tensor_from_f32(client, shape, vec![0.0; num_elements])
}

fn tensor_from_f32(
    client: &ComputeClient<CudaRuntime>,
    shape: Shape,
    values: Vec<f32>,
) -> TensorHandle<CudaRuntime> {
    let values = values.into_iter().map(f16::from_f32).collect::<Vec<f16>>();
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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| (*actual - *expected).abs() > TOLERANCE);
    assert!(
        mismatch.is_none(),
        "Parity mismatch at {:?}; tolerance={TOLERANCE}",
        mismatch.map(|(index, (actual, expected))| (index, actual, expected)),
    );
}

fn is_valid_input_position(
    case: &ConvolutionCase,
    out_y: usize,
    out_x: usize,
    ky: usize,
    kx: usize,
) -> bool {
    let in_y = out_y as isize * case.stride[0] as isize + ky as isize * case.dilation[0] as isize
        - case.padding[0] as isize;
    let in_x = out_x as isize * case.stride[1] as isize + kx as isize * case.dilation[1] as isize
        - case.padding[1] as isize;

    in_y >= 0 && in_y < case.in_h as isize && in_x >= 0 && in_x < case.in_w as isize
}

fn expected_forward(case: &ConvolutionCase) -> Vec<f32> {
    let mut output =
        Vec::with_capacity(case.batches * case.out_h() * case.out_w() * case.out_channels);
    let bias = usize::from(case.has_bias);

    for _ in 0..case.batches {
        for out_y in 0..case.out_h() {
            for out_x in 0..case.out_w() {
                let kernel_positions = (0..case.kernel_size[0])
                    .flat_map(|ky| (0..case.kernel_size[1]).map(move |kx| (ky, kx)))
                    .filter(|(ky, kx)| is_valid_input_position(case, out_y, out_x, *ky, *kx))
                    .count();
                let expected = (kernel_positions * case.in_channels) as f32
                    * FORWARD_INPUT_VALUE
                    * FORWARD_WEIGHT_VALUE
                    + bias as f32;
                output.extend(std::iter::repeat_n(expected, case.out_channels));
            }
        }
    }

    output
}

fn expected_backward_data(case: &ConvolutionCase) -> Vec<f32> {
    let mut output = vec![0.0; case.batches * case.in_h * case.in_w * case.in_channels];

    for batch in 0..case.batches {
        for out_y in 0..case.out_h() {
            for out_x in 0..case.out_w() {
                for ky in 0..case.kernel_size[0] {
                    for kx in 0..case.kernel_size[1] {
                        if !is_valid_input_position(case, out_y, out_x, ky, kx) {
                            continue;
                        }
                        let in_y = out_y * case.stride[0] + ky * case.dilation[0] - case.padding[0];
                        let in_x = out_x * case.stride[1] + kx * case.dilation[1] - case.padding[1];
                        let offset =
                            ((batch * case.in_h + in_y) * case.in_w + in_x) * case.in_channels;
                        for value in &mut output[offset..offset + case.in_channels] {
                            *value += case.out_channels as f32;
                        }
                    }
                }
            }
        }
    }

    output
}

fn expected_backward_weight(case: &ConvolutionCase) -> Vec<f32> {
    let mut output =
        vec![0.0; case.out_channels * case.kernel_size[0] * case.kernel_size[1] * case.in_channels];

    for ky in 0..case.kernel_size[0] {
        for kx in 0..case.kernel_size[1] {
            let spatial_positions = (0..case.out_h())
                .flat_map(|out_y| (0..case.out_w()).map(move |out_x| (out_y, out_x)))
                .filter(|(out_y, out_x)| is_valid_input_position(case, *out_y, *out_x, ky, kx))
                .count();
            let expected = (case.batches * spatial_positions) as f32
                * WGRAD_INPUT_VALUE
                * WGRAD_OUT_GRAD_VALUE;

            for out_channel in 0..case.out_channels {
                let offset = ((out_channel * case.kernel_size[0] + ky) * case.kernel_size[1] + kx)
                    * case.in_channels;
                output[offset..offset + case.in_channels].fill(expected);
            }
        }
    }

    output
}

fn assert_forward_parity(
    case: &ConvolutionCase,
    algorithm: ConvAlgorithm,
    tile_kind: AcceleratedTileKind,
) {
    let client = CudaRuntime::client(&Default::default());
    let dtypes = f16_dtypes();
    let input = filled_tensor(&client, input_shape(case), FORWARD_INPUT_VALUE);
    let weight = filled_tensor(&client, weight_shape(case), FORWARD_WEIGHT_VALUE);
    let bias = case
        .has_bias
        .then(|| ones_tensor(&client, shape![case.out_channels]));
    let expected = expected_forward(case);

    for _ in 0..REPETITIONS {
        let output = zeros_tensor(&client, output_shape(case));
        launch_ref(
            &strategy(algorithm, tile_kind),
            &client,
            ConvolutionInputs::Forward {
                input: InputBinding::new(input.clone().binding(), dtypes.lhs_global),
                weight: InputBinding::new(weight.clone().binding(), dtypes.rhs_global),
                bias: bias
                    .clone()
                    .map(|bias| InputBinding::new(bias.binding(), dtypes.lhs_global)),
                out: output.clone().binding(),
            },
            args(case),
            dtypes.clone(),
        )
        .unwrap();
        assert_close(&read_f16(&client, output), &expected);
    }
}

fn assert_backward_data_parity(
    case: &ConvolutionCase,
    algorithm: ConvAlgorithm,
    tile_kind: AcceleratedTileKind,
) {
    let client = CudaRuntime::client(&Default::default());
    let dtypes = f16_dtypes();
    let out_grad = ones_tensor(&client, output_shape(case));
    let weight = ones_tensor(&client, weight_shape(case));
    let expected = expected_backward_data(case);

    for _ in 0..REPETITIONS {
        let in_grad = zeros_tensor(&client, input_shape(case));
        launch_ref(
            &strategy(algorithm, tile_kind),
            &client,
            ConvolutionInputs::BackwardData {
                out_grad: InputBinding::new(out_grad.clone().binding(), dtypes.lhs_global),
                weights: InputBinding::new(weight.clone().binding(), dtypes.rhs_global),
                in_grad: in_grad.clone().binding(),
            },
            args(case),
            dtypes.clone(),
        )
        .unwrap();
        assert_close(&read_f16(&client, in_grad), &expected);
    }
}

fn assert_backward_weight_parity(
    case: &ConvolutionCase,
    algorithm: ConvAlgorithm,
    tile_kind: AcceleratedTileKind,
) {
    let client = CudaRuntime::client(&Default::default());
    let dtypes = f16_dtypes();
    let input = filled_tensor(&client, input_shape(case), WGRAD_INPUT_VALUE);
    let out_grad = filled_tensor(&client, output_shape(case), WGRAD_OUT_GRAD_VALUE);
    let expected = expected_backward_weight(case);

    for _ in 0..REPETITIONS {
        let weight_grad = zeros_tensor(&client, weight_shape(case));
        launch_ref(
            &strategy(algorithm, tile_kind),
            &client,
            ConvolutionInputs::BackwardWeight {
                input: InputBinding::new(input.clone().binding(), dtypes.lhs_global),
                out_grad: InputBinding::new(out_grad.clone().binding(), dtypes.rhs_global),
                weight_grad: weight_grad.clone().binding(),
            },
            args(case),
            dtypes.clone(),
        )
        .unwrap();
        assert_close(&read_f16(&client, weight_grad), &expected);
    }
}

#[test]
fn test_terminalo3_forward_simple_async_tma_mma_parity() {
    assert_forward_parity(
        &rollout_case(),
        ConvAlgorithm::SimpleAsyncTma,
        AcceleratedTileKind::Mma,
    );
}

#[test]
fn test_terminalo3_forward_simple_async_tma_mma_without_bias_parity() {
    let mut case = rollout_case();
    case.has_bias = false;
    assert_forward_parity(
        &case,
        ConvAlgorithm::SimpleAsyncTma,
        AcceleratedTileKind::Mma,
    );
}

#[test]
fn test_terminalo3_backward_weight_simple_async_tma_cmma_parity() {
    assert_backward_weight_parity(
        &learner_cmma_case(),
        ConvAlgorithm::SimpleAsyncTma,
        AcceleratedTileKind::Cmma,
    );
}

#[test]
fn test_terminalo3_backward_weight_simple_async_tma_mma_parity() {
    assert_backward_weight_parity(
        &rollout_case(),
        ConvAlgorithm::SimpleAsyncTma,
        AcceleratedTileKind::Mma,
    );
}

#[test]
fn test_terminalo3_backward_data_simple_async_tma_mma_parity() {
    assert_backward_data_parity(
        &rollout_case(),
        ConvAlgorithm::SimpleAsyncTma,
        AcceleratedTileKind::Mma,
    );
}

#[test]
fn test_terminalo3_forward_non_tma_implicit_gemm_parity() {
    let case = compact_case();
    for (algorithm, tile_kind) in [
        (ConvAlgorithm::SimpleSyncCyclic, AcceleratedTileKind::Cmma),
        (ConvAlgorithm::SimpleSyncStrided, AcceleratedTileKind::Mma),
        (ConvAlgorithm::SimpleAsyncCyclic, AcceleratedTileKind::Cmma),
        (ConvAlgorithm::SimpleAsyncStrided, AcceleratedTileKind::Mma),
    ] {
        assert_forward_parity(&case, algorithm, tile_kind);
    }
}

#[test]
fn test_terminalo3_backward_data_non_tma_implicit_gemm_parity() {
    let case = compact_case();
    for (algorithm, tile_kind) in [
        (ConvAlgorithm::SimpleSyncCyclic, AcceleratedTileKind::Cmma),
        (ConvAlgorithm::SimpleSyncStrided, AcceleratedTileKind::Mma),
        (ConvAlgorithm::SimpleAsyncCyclic, AcceleratedTileKind::Cmma),
        (ConvAlgorithm::SimpleAsyncStrided, AcceleratedTileKind::Mma),
    ] {
        assert_backward_data_parity(&case, algorithm, tile_kind);
    }
}

#[test]
fn test_terminalo3_backward_weight_non_tma_implicit_gemm_parity() {
    let case = compact_case();
    for (algorithm, tile_kind) in [
        (ConvAlgorithm::SimpleSyncCyclic, AcceleratedTileKind::Cmma),
        (ConvAlgorithm::SimpleSyncStrided, AcceleratedTileKind::Mma),
        (ConvAlgorithm::SimpleAsyncCyclic, AcceleratedTileKind::Cmma),
        (ConvAlgorithm::SimpleAsyncStrided, AcceleratedTileKind::Mma),
    ] {
        assert_backward_weight_parity(&case, algorithm, tile_kind);
    }
}
