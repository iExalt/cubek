use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::std::tensor::{
    View, ViewMut,
    launch::{ScaleBindings, ViewArg},
    layout::{Coords1d, VirtualLayout, VirtualLayoutLaunch},
};
use cubecl::unexpanded;
use cubek_std::{InputBinding, MatrixLayout, launch::tma::tma_operand};

pub use crate::routine::RuntimeConfig;
use crate::{
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
    multi_level::{
        BatchMatmulRoutine,
        components::global::memory::{
            BatchLayout, BatchLayoutLaunch, GlobalLayout, GlobalLayoutConfig, GlobalLayoutLaunch,
            GlobalScaleLayout, NoopLayout, NoopLayoutLaunch, SimpleTmaGlobalLayout,
            SimpleTmaGlobalLayoutLaunch,
        },
        definition::Blueprint,
        stage::{SwizzleMode, SwizzleModes},
    },
};

define_scalar!(pub Lhs);
define_scalar!(pub Rhs);
define_scalar!(pub Acc);

define_size!(pub LhsSize);
define_size!(pub RhsSize);
define_size!(pub AccSize);

/// Input argument
pub type InputArg<MA> =
    <MA as MatmulArgs>::Input<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>;

/// Output argument
pub type OutputArg<MA> = <MA as MatmulArgs>::Output<Vector<Acc, AccSize>>;

/// Config argument
pub type ConfigArg<MA> = <MA as MatmulArgs>::Config;

/// Input runtime argument
pub type InputRuntimeArg<MA> = <InputArg<MA> as LaunchArg>::RuntimeArg;

/// Config runtime argument
pub type ConfigRuntimeArg<MA> = <ConfigArg<MA> as LaunchArg>::RuntimeArg;

/// Output runtime argument
pub type OutputRuntimeArg<MA> = <OutputArg<MA> as LaunchArg>::RuntimeArg;

pub type BatchedCoords = (usize, u32, u32);

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory<A: BatchMatmulRoutine<()>>: LaunchArg {
    fn validate(
        _blueprint: &A::Blueprint,
        _problem: &MatmulProblem,
        _vector_sizes: &MatmulVectorSizes,
        _dtypes: &MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        lhs: InputBinding,
        rhs: InputBinding,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        vector_sizes: &MatmulVectorSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg;
}

/// Create the output runtime argument for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory<A: BatchMatmulRoutine<()>>: LaunchArg {
    #[allow(clippy::too_many_arguments)]
    fn create(
        out: TensorBinding,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        vector_sizes: &MatmulVectorSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg;
}

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>: LaunchArg + CubeType;

    /// Type used for the output.
    type Output<EO: CubePrimitive>: LaunchArg + CubeType;

    /// Type used for runtime configuration.
    type Config: RuntimeConfig;

    /// Inner state that is used to create tensor inputs and
    /// tensor outputs.
    type State<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>: CubeType;

    /// Init the state.
    fn init_state<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] lhs_layout_config: GlobalLayoutConfig,
        #[comptime] rhs_layout_config: GlobalLayoutConfig,
        #[comptime] out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO>;

    fn view_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<'_, Lhs, BatchedCoords> {
        unexpanded!()
    }
    fn batch_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }
    fn view_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<'_, Rhs, BatchedCoords> {
        unexpanded!()
    }
    fn batch_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }
    fn view_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<'_, EO, BatchedCoords>> {
        unexpanded!()
    }
    fn batch_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }
    fn view_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> ViewMut<'_, EO, BatchedCoords> {
        unexpanded!()
    }
    fn batch_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _batch: usize,
    ) -> usize {
        unexpanded!()
    }

    fn runtime_config<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        unexpanded!()
    }
}

#[derive(Clone, Copy)]
/// Identification of the tensor input.
pub enum TensorInputIdent {
    Lhs,
    Rhs,
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorArgs<Config: RuntimeConfig = ()> {
    _config: PhantomData<Config>,
}

#[derive(CubeLaunch, CubeType, Clone)]
#[expand(derive(Clone))]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<Lhs: CubePrimitive, Rhs: CubePrimitive, Acc: CubePrimitive> {
    /// The lhs tensor.
    lhs_batch: VirtualLayout<Coords1d, Coords1d>,
    lhs: View<'static, Lhs, BatchedCoords>,
    /// The rhs tensor.
    rhs_batch: VirtualLayout<Coords1d, Coords1d>,
    rhs: View<'static, Rhs, BatchedCoords>,
    /// The tensor for loading the accumulator, if present
    acc_batch: ComptimeOption<VirtualLayout<Coords1d, Coords1d>>,
    acc: ComptimeOption<View<'static, Acc, BatchedCoords>>,
}

impl<Lhs: CubePrimitive, Rhs: CubePrimitive, Acc: CubePrimitive, A: BatchMatmulRoutine<()>>
    ConcreteInputsFactory<A> for TensorInputs<Lhs, Rhs, Acc>
{
    fn create(
        lhs: InputBinding,
        rhs: InputBinding,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        vector_sizes: &MatmulVectorSizes,
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg {
        let view = |handle: InputBinding, config: GlobalLayoutConfig, vector_size| match handle {
            InputBinding::Normal(handle, _dtype) => {
                let layout = GlobalLayoutLaunch::from_handle(&handle, vector_size, config);
                ViewArg::new_tensor::<GlobalLayout>(handle.into_tensor_arg(), layout)
            }
            InputBinding::Quantized {
                data,
                scale,
                shape,
                scheme,
                ..
            } => {
                let (data_layout, scales_layout) = GlobalLayoutLaunch::from_quantized_handle(
                    &data,
                    &scale,
                    &shape,
                    problem,
                    scheme,
                    vector_size,
                    config,
                );
                let data_view =
                    ViewArg::new_tensor::<GlobalLayout>(data.into_tensor_arg(), data_layout);
                let scales_view = ViewArg::new_tensor::<GlobalScaleLayout>(
                    scale.into_tensor_arg(),
                    scales_layout,
                );
                ViewArg::new_quantized(data_view, ScaleBindings::one(scales_view), scheme)
            }
        };
        let batch_layout = |handle: &InputBinding| match handle {
            InputBinding::Normal(handle, _dtype) => {
                let layout = BatchLayoutLaunch::from_handle(handle, problem);
                VirtualLayoutLaunch::new::<BatchLayout>(layout)
            }
            InputBinding::Quantized { .. } => {
                VirtualLayoutLaunch::new::<NoopLayout>(NoopLayoutLaunch::new())
            }
        };

        TensorInputsLaunch::new(
            batch_layout(&lhs),
            view(lhs, blueprint.lhs_global_layout_config(), vector_sizes.lhs),
            batch_layout(&rhs),
            view(rhs, blueprint.rhs_global_layout_config(), vector_sizes.rhs),
            ComptimeOptionArgs::None,
            ComptimeOptionArgs::None,
        )
    }
}

#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
pub struct TensorOutput<EG: CubePrimitive> {
    view: ViewMut<'static, EG, BatchedCoords>,
    batch: VirtualLayout<Coords1d, Coords1d>,
}

impl<EG: CubePrimitive, A: BatchMatmulRoutine<()>> ConcreteOutputFactory<A> for TensorOutput<EG> {
    fn create(
        out: TensorBinding,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        vector_sizes: &MatmulVectorSizes,
        _dtypes: &MatmulElems,
    ) -> Self::RuntimeArg {
        let layout = GlobalLayoutLaunch::from_handle(
            &out,
            vector_sizes.out,
            blueprint.out_global_layout_config(),
        );
        let batch = BatchLayoutLaunch::from_handle(&out, problem);
        let view = ViewArg::new_tensor::<GlobalLayout>(out.into_tensor_arg(), layout);
        TensorOutputLaunch::new(view, VirtualLayoutLaunch::new::<BatchLayout>(batch))
    }
}

#[cube]
impl<Config: RuntimeConfig> MatmulArgs for TensorArgs<Config> {
    type Output<EO: CubePrimitive> = TensorOutput<EO>;
    type Input<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> =
        TensorInputs<Lhs, Rhs, EO>;
    type Config = Config;
    type State<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> =
        (TensorInputs<Lhs, Rhs, EO>, TensorOutput<EO>, Config);

    fn init_state<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] _lhs_layout_config: GlobalLayoutConfig,
        #[comptime] _rhs_layout_config: GlobalLayoutConfig,
        #[comptime] _out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO> {
        (input.clone(), output.clone(), config)
    }

    fn view_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<'_, Lhs, BatchedCoords> {
        state.0.lhs
    }

    fn batch_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.0.lhs_batch.to_source_pos(batch)
    }

    fn view_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<'_, Rhs, BatchedCoords> {
        state.0.rhs
    }

    fn batch_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.0.rhs_batch.to_source_pos(batch)
    }

    fn view_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<'_, EO, BatchedCoords>> {
        state.0.acc.map(|view| view) // Lifetime coercion hack
    }

    fn batch_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        #[comptime]
        match &state.0.acc_batch {
            ComptimeOption::Some(layout) => layout.to_source_pos(batch),
            ComptimeOption::None => batch,
        }
    }

    fn view_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ViewMut<'_, EO, BatchedCoords> {
        state.1.view
    }

    fn batch_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.1.batch.to_source_pos(batch)
    }

    fn runtime_config<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        state.2.clone()
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensor maps.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorMapArgs<Config: RuntimeConfig = ()> {
    _config: PhantomData<Config>,
}

#[derive(CubeLaunch, CubeType, Clone)]
#[expand(derive(Clone))]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorMapInputs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> {
    /// The lhs tensor.
    pub lhs: View<'static, Lhs, BatchedCoords>,
    /// The rhs tensor.
    pub rhs: View<'static, Rhs, BatchedCoords>,
    /// The accumulator
    pub acc: ComptimeOption<View<'static, EO, BatchedCoords>>,
    /// The accumulator batch layout
    pub acc_batch: ComptimeOption<VirtualLayout<Coords1d, Coords1d>>,
}

impl<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive, A: BatchMatmulRoutine<()>>
    ConcreteInputsFactory<A> for TensorMapInputs<Lhs, Rhs, EO>
{
    fn validate(
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        _vector_sizes: &MatmulVectorSizes,
        _dtypes: &MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        let (lhs, rhs) = tma_tile_shapes(blueprint, problem);
        validate_tma_tile_shape("lhs", lhs)?;
        validate_tma_tile_shape("rhs", rhs)
    }

    fn create(
        lhs_handle: InputBinding,
        rhs_handle: InputBinding,
        blueprint: &A::Blueprint,
        problem: &MatmulProblem,
        _vector_sizes: &MatmulVectorSizes,
        dtypes: &MatmulElems,
    ) -> Self::RuntimeArg {
        let lhs = lhs_handle.into_data();
        let rhs = rhs_handle.into_data();

        let (box_lhs, box_rhs) = tma_tile_shapes(blueprint, problem);

        // Logical (batches, rows, cols), read before `tma_operand` consumes the bindings.
        let dims = |binding: &TensorBinding, batches: &[usize]| {
            let rank = binding.shape.len();
            (
                batches.iter().product::<usize>(),
                binding.shape[rank - 2] as u32,
                binding.shape[rank - 1] as u32,
            )
        };
        let lhs_dims = dims(&lhs, &problem.lhs_batches);
        let rhs_dims = dims(&rhs, &problem.rhs_batches);

        let (lhs, lhs_transposed) = tma_operand(
            lhs,
            lhs_dims.0,
            problem.lhs_layout,
            box_lhs,
            dtypes.lhs_stage,
            blueprint.swizzle_modes().lhs.into(),
        );
        let (rhs, rhs_transposed) = tma_operand(
            rhs,
            rhs_dims.0,
            problem.rhs_layout,
            box_rhs,
            dtypes.rhs_stage,
            blueprint.swizzle_modes().rhs.into(),
        );

        let view = |buffer, shape: (usize, u32, u32), transposed| {
            let layout = SimpleTmaGlobalLayoutLaunch::new(transposed, shape);
            ViewArg::new_tensor_map_tiled::<SimpleTmaGlobalLayout>(buffer, layout)
        };

        TensorMapInputsLaunch::new(
            view(lhs, lhs_dims, lhs_transposed),
            view(rhs, rhs_dims, rhs_transposed),
            ComptimeOptionArgs::None,
            ComptimeOptionArgs::None,
        )
    }
}

fn tma_tile_shapes<B: Blueprint>(
    blueprint: &B,
    problem: &MatmulProblem,
) -> ((usize, usize), (usize, usize)) {
    tma_box_shapes(
        blueprint.tiling_scheme(),
        blueprint.swizzle_modes(),
        problem.lhs_layout,
        problem.rhs_layout,
    )
}

fn tma_box_shapes(
    tiling_scheme: crate::multi_level::definition::TilingScheme,
    swizzle_modes: SwizzleModes,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
) -> ((usize, usize), (usize, usize)) {
    let stage_m = tiling_scheme.elements_per_stage_along_m() as usize;
    let stage_n = tiling_scheme.elements_per_stage_along_n() as usize;
    let stage_k = tiling_scheme.elements_per_stage_along_k() as usize;
    let (tile_m, tile_n, tile_k) = (
        tiling_scheme.tile_size.m as usize,
        tiling_scheme.tile_size.n as usize,
        tiling_scheme.tile_size.k as usize,
    );

    // Boxes are logical (rows, cols); tma_operand puts them in descriptor order.
    let lhs = match swizzle_modes.lhs {
        SwizzleMode::None => match lhs_layout {
            MatrixLayout::RowMajor => (stage_m, tile_k),
            MatrixLayout::ColMajor => (tile_m, stage_k),
        },
        _ => (stage_m, stage_k),
    };
    let rhs = match swizzle_modes.rhs {
        SwizzleMode::None => match rhs_layout {
            MatrixLayout::RowMajor => (stage_k, tile_n),
            MatrixLayout::ColMajor => (tile_k, stage_n),
        },
        _ => (stage_k, stage_n),
    };
    (lhs, rhs)
}

fn validate_tma_tile_shape(
    name: &str,
    (rows, cols): (usize, usize),
) -> Result<(), MatmulSetupError> {
    if !(1..=256).contains(&rows) || !(1..=256).contains(&cols) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "{name} TMA tile shape ({rows}, {cols}) must have dimensions in 1..=256"
        ))));
    }
    Ok(())
}

#[cube]
impl<Config: RuntimeConfig> MatmulArgs for TensorMapArgs<Config> {
    type Input<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> =
        TensorMapInputs<Lhs, Rhs, EO>;
    type Output<EO: CubePrimitive> = TensorOutput<EO>;
    type Config = Config;
    type State<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> =
        (TensorMapInputs<Lhs, Rhs, EO>, TensorOutput<EO>, Config);

    fn init_state<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] _lhs_layout_config: GlobalLayoutConfig,
        #[comptime] _rhs_layout_config: GlobalLayoutConfig,
        #[comptime] _out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO> {
        (input.clone(), output.clone(), config)
    }

    fn view_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<'_, Lhs, BatchedCoords> {
        state.0.lhs
    }

    fn batch_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        batch
    }

    fn view_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<'_, Rhs, BatchedCoords> {
        state.0.rhs
    }

    fn batch_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        _state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        batch
    }

    fn view_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<'_, EO, BatchedCoords>> {
        state.0.acc.map(|view| view) // Lifetime coercion hack
    }

    fn batch_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        #[comptime]
        #[comptime]
        match &state.0.acc_batch {
            ComptimeOption::Some(layout) => layout.to_source_pos(batch),
            ComptimeOption::None => batch,
        }
    }

    fn view_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ViewMut<'_, EO, BatchedCoords> {
        state.1.view
    }

    fn batch_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        state.1.batch.to_source_pos(batch)
    }

    fn runtime_config<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        state.2.clone()
    }
}

#[cfg(test)]
mod tma_tests {
    use super::*;

    fn tiling_scheme() -> crate::multi_level::definition::TilingScheme {
        crate::multi_level::definition::TilingScheme::builder()
            .with_tile_size((8, 16, 4).into())
            .with_partition_size((2, 2, 2).into())
            .with_stage_size((2, 2, 1).into())
            .build()
            .unwrap()
    }

    #[test]
    fn tma_box_shapes_follow_layout_and_swizzle() {
        let none = SwizzleModes {
            lhs: SwizzleMode::None,
            rhs: SwizzleMode::None,
            ..Default::default()
        };
        assert_eq!(
            tma_box_shapes(
                tiling_scheme(),
                none,
                MatrixLayout::RowMajor,
                MatrixLayout::RowMajor,
            ),
            ((32, 4), (8, 16))
        );

        assert_eq!(
            tma_box_shapes(
                tiling_scheme(),
                none,
                MatrixLayout::ColMajor,
                MatrixLayout::ColMajor,
            ),
            ((8, 8), (4, 64))
        );

        let swizzled = SwizzleModes {
            lhs: SwizzleMode::B32,
            rhs: SwizzleMode::B64,
            ..Default::default()
        };
        assert_eq!(
            tma_box_shapes(
                tiling_scheme(),
                swizzled,
                MatrixLayout::ColMajor,
                MatrixLayout::RowMajor,
            ),
            ((32, 8), (8, 64))
        );
    }

    #[test]
    fn tma_tile_shape_validation_enforces_descriptor_bounds() {
        for (rows, cols) in [(1, 1), (256, 256), (1, 256), (256, 1)] {
            assert!(validate_tma_tile_shape("operand", (rows, cols)).is_ok());
        }
        for shape in [(0, 1), (1, 0), (257, 1), (1, 257), (257, 257)] {
            let error = validate_tma_tile_shape("operand", shape).unwrap_err();
            assert!(format!("{error:?}").contains("operand TMA tile shape"));
        }
    }
}
