//! Forward convolution bias regression.
//!
//! The shared convolution harness supplies no bias, so it cannot catch a
//! kernel that computes the convolution correctly while dropping the bias
//! accumulator.

use cubecl::{frontend::Scalar, prelude::*, zspace::shape};
use cubek_convolution::{
    ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, components::ConvSetupError,
    launch_ref,
};
use cubek_matmul::definition::{MatmulElems, MatmulSetupError};
use cubek_std::InputBinding;
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, ValidationResult,
    launch_and_capture_outcome,
};

#[test]
fn simple_sync_cyclic_cmma_forward_bias_f16() {
    let client = cubecl::test_device().client();
    let f16 = half::f16::elem_type_native();
    let dtypes = MatmulElems::from_globals(&cubek_matmul::definition::MatmulGlobalElems {
        lhs: f16,
        rhs: f16,
        out: f16,
    });

    let input = TestInput::builder(client.clone(), shape![1, 4, 8, 16])
        .dtype(dtypes.lhs_global)
        .custom(vec![1.; 512])
        .generate_without_host_data();
    let weight = TestInput::builder(client.clone(), shape![32, 1, 1, 16])
        .dtype(dtypes.rhs_global)
        .custom(vec![1.; 512])
        .generate_without_host_data();
    let (bias, _) = TestInput::builder(client.clone(), shape![32])
        .dtype(dtypes.acc_global)
        .custom((1..=32).map(|value| value as f32).collect())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![1, 4, 8, 32])
        .dtype(dtypes.acc_global)
        .zeros()
        .generate_without_host_data();

    let inputs = ConvolutionInputs::Forward {
        input: InputBinding::new(input.binding(), dtypes.lhs_global),
        weight: InputBinding::new(weight.binding(), dtypes.rhs_global),
        bias: Some(InputBinding::new(bias.binding(), dtypes.acc_global)),
        out: out.clone().binding(),
    };
    let args = ConvolutionArgs::<2> {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
    };
    let strategy = Strategy::Inferred {
        algorithm: ConvAlgorithm::SimpleSyncCyclic,
        tile_kind: cubek_convolution::AcceleratedTileKind::Cmma,
    };

    let is_cpu = client.properties().hardware.num_cpu_cores.is_some();
    let mut unavailable = None;
    let outcome = launch_and_capture_outcome(&client, &[&out.handle], |client| {
        match launch_ref(&strategy, client, inputs, args, dtypes.clone()) {
            Ok(()) => ExecutionOutcome::Executed,
            Err(error @ ConvSetupError::Matmul(MatmulSetupError::Unavailable(_))) => {
                let reason = format!("{error:?}");
                if is_cpu {
                    unavailable = Some(reason.clone());
                    ExecutionOutcome::CompileError(reason)
                } else {
                    panic!("native/non-CPU forward-bias launch was unavailable: {reason}");
                }
            }
            Err(error) => panic!("forward-bias launch failed: {error:?}"),
        }
    });

    if let Some(reason) = unavailable {
        eprintln!("forward-bias CPU skipped: {reason}");
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "CPU SimpleSyncCyclic Cmma is unavailable: {reason}"
        )))
        .enforce();
        return;
    }

    match outcome {
        ExecutionOutcome::Executed => {
            let actual = HostData::from_tensor_handle(&client, out, HostDataType::F32);
            for row in 0..4 {
                for column in 0..8 {
                    for channel in 0..32 {
                        let expected = 16. + (channel + 1) as f32;
                        assert_eq!(
                            actual.get_f32(&[0, row, column, channel]),
                            expected,
                            "bias mismatch at row {row}, column {column}, channel {channel}",
                        );
                    }
                }
            }
        }
        ExecutionOutcome::CompileError(error) => {
            panic!("forward-bias launch did not execute: {error}");
        }
    }
}
