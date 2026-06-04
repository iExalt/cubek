use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::{
    stage::StageMemoryConfig,
    tile::{StridedTile, TilingValidation},
    {InvalidConfigError, MatrixLayout},
};

use crate::components::stage::bias_stage::BiasStageMemory;

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl BiasTilingLayout {
    pub fn get_tile<ES: Numeric, NS: Size>(
        stage: &BiasStageMemory<ES, NS>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES, NS> {
        let (row, col) = tile;
        let stage_vector_size = config.vector_size;
        let matrix_layout = config.matrix_layout;

        match matrix_layout {
            MatrixLayout::RowMajor => {
                let tile_size_row = config.elements_per_tile_along_row;
                let tile_size_col = config.elements_per_tile_along_col / stage_vector_size;
                let stride = config.elements_per_stage_along_col() / stage_vector_size;
                let length = (tile_size_row - 1) * stride + tile_size_col;
                let start = row * tile_size_row * stride + col * tile_size_col;

                StridedTile::new_strided(
                    stage.as_slice(),
                    start,
                    start + length,
                    stride,
                    stage.swizzle,
                    matrix_layout,
                )
            }
            MatrixLayout::ColMajor => {
                let tile_size_row = config.elements_per_tile_along_row / stage_vector_size;
                let tile_size_col = config.elements_per_tile_along_col;
                let stride = config.elements_per_stage_along_row() / stage_vector_size;
                let length = (tile_size_col - 1) * stride + tile_size_row;
                let start = row * tile_size_row + col * tile_size_col * stride;

                StridedTile::new_strided(
                    stage.as_slice(),
                    start,
                    start + length,
                    stride,
                    stage.swizzle,
                    matrix_layout,
                )
            }
        }
    }
}

impl TilingValidation for BiasTilingLayout {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        let stage_width = config.elements_per_stage_along_col();
        if config.vector_size > stage_width {
            return Err(Box::new(format!(
                "Invalid vector size. Got {:?} which should not be >{:?}",
                config.vector_size, stage_width,
            )));
        }
        Ok(())
    }
}
