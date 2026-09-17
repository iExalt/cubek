use cubecl::{
    features::TypeUsage,
    ir::{ElemType, FloatKind},
    prelude::*,
    tensor_vector_size_parallel,
    zspace::shape,
};
use cubek_convolution::{ConvolutionArgs, DirectTensors, launch_direct};
use cubek_test_utils::{HostData, HostDataType, TestInput};

fn direct_result(
    dtype: ElemType,
    input_shape: impl Into<cubecl::zspace::Shape>,
    input: Vec<f32>,
    weight_shape: impl Into<cubecl::zspace::Shape>,
    weight: Vec<f32>,
) -> f32 {
    let client = cubecl::test_device().client();
    let input = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .custom(input)
        .generate_without_host_data();
    let weight = TestInput::builder(client.clone(), weight_shape)
        .dtype(dtype)
        .custom(weight)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![1, 1, 1, 1])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    launch_direct(
        &client,
        DirectTensors {
            input: input.binding(),
            weight: weight.binding(),
            bias: None,
            out: out.clone().binding(),
        },
        ConvolutionArgs {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        },
        1,
        dtype,
    )
    .unwrap();

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
        .data
        .get_f32(0)
}

#[test]
fn direct_f16_accumulates_spatial_reduction_in_f32() {
    // At 2048, an F16 ulp is 2, so adding 1 repeatedly to a half accumulator
    // stalls. The promoted accumulator reaches 3071 before the final F16 cast,
    // which rounds to the exactly representable 3072.
    let mut input = vec![1.0; 1024];
    input[0] = 2048.0;
    assert_eq!(
        direct_result(
            ElemType::Float(FloatKind::F16),
            shape![1, 1024, 1, 1],
            input,
            shape![1, 1024, 1, 1],
            vec![1.0; 1024],
        ),
        3072.0
    );
}

#[test]
fn direct_f16_accumulates_each_vector_lane_in_f32() {
    let client = cubecl::test_device().client();
    let dtype = ElemType::Float(FloatKind::F16);
    let channels = 16384;
    let weight_shape = shape![1, 1, 1, channels];
    let vector_size_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &weight_shape,
        &vec![channels, channels, channels, 1].into(),
        3,
    );
    if client.properties().hardware.plane_size_max != 1 || vector_size_in <= 1 {
        eprintln!(
            "SKIP direct F16 lane accumulation: CPU lane path requires plane size 1 and vector size > 1 (plane={}, vector={vector_size_in})",
            client.properties().hardware.plane_size_max,
        );
        return;
    }

    let mut input = vec![1.0; channels];
    input[0] = 2048.0;
    // The lane path accumulates one vector lane independently before folding
    // lanes into the output channel. With EA=F32, the sum is 18431 and the
    // final F16 cast is 18432; an E=F16 accumulator loses the first lane's
    // low-order contributions and produces a smaller value.
    assert_eq!(
        direct_result(
            dtype,
            shape![1, 1, 1, channels],
            input,
            weight_shape,
            vec![1.0; channels],
        ),
        18432.0
    );
}

#[test]
fn direct_bf16_accumulates_spatial_reduction_in_f32_when_supported() {
    let client = cubecl::test_device().client();
    let bf16_uses = half::bf16::supported_uses(&client);
    if !bf16_uses.contains(TypeUsage::Conversion) || !bf16_uses.contains(TypeUsage::Buffer) {
        eprintln!("SKIP direct BF16 accumulation: runtime lacks BF16 conversion or buffer support");
        return;
    }

    let mut input = vec![1.0; 1024];
    input[0] = 2048.0;
    let actual = direct_result(
        ElemType::Float(FloatKind::BF16),
        shape![1, 1024, 1, 1],
        input,
        shape![1, 1024, 1, 1],
        vec![1.0; 1024],
    );

    assert_eq!(actual, 3072.0);
}
