//! Minimal dense 2D array with the padding helpers domain growth needs.

use std::ops::{Index, IndexMut};

/// A dense, row-major `rows x cols` 2D array. Element `(row, col)` lives at
/// flat index `row * cols + col`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid2<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Grid2<T> {
    /// A `rows x cols` grid with every cell set to `value`.
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Grid2 {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Wrap a row-major `data` vector; panics if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "grid data length mismatch");
        Grid2 { rows, cols, data }
    }

    /// Grow to `new_shape`, placing the old content at `(row_shift,
    /// col_shift)` and filling the border by replicating old edge values
    /// (numpy `pad(mode="edge")`).
    pub fn pad_edge(
        &self,
        new_shape: (usize, usize),
        row_shift: usize,
        col_shift: usize,
    ) -> Grid2<T> {
        let (new_rows, new_cols) = new_shape;
        assert!(row_shift + self.rows <= new_rows);
        assert!(col_shift + self.cols <= new_cols);
        let mut data = Vec::with_capacity(new_rows * new_cols);
        for row in 0..new_rows {
            let src_row = row
                .saturating_sub(row_shift)
                .min(self.rows - 1);
            for col in 0..new_cols {
                let src_col = col
                    .saturating_sub(col_shift)
                    .min(self.cols - 1);
                data.push(self[(src_row, src_col)].clone());
            }
        }
        Grid2::from_vec(new_rows, new_cols, data)
    }

    /// Grow to `new_shape` filling new cells with `fill` (numpy
    /// `pad(mode="constant")`).
    pub fn pad_const(
        &self,
        new_shape: (usize, usize),
        row_shift: usize,
        col_shift: usize,
        fill: T,
    ) -> Grid2<T> {
        let (new_rows, new_cols) = new_shape;
        assert!(row_shift + self.rows <= new_rows);
        assert!(col_shift + self.cols <= new_cols);
        let mut out = Grid2::filled(new_rows, new_cols, fill);
        for row in 0..self.rows {
            for col in 0..self.cols {
                out[(row + row_shift, col + col_shift)] =
                    self[(row, col)].clone();
            }
        }
        out
    }

    /// Copy `other` into this grid at `(row_shift, col_shift)`.
    pub fn blit(&mut self, other: &Grid2<T>, row_shift: usize, col_shift: usize) {
        assert!(row_shift + other.rows <= self.rows);
        assert!(col_shift + other.cols <= self.cols);
        for row in 0..other.rows {
            for col in 0..other.cols {
                self[(row + row_shift, col + col_shift)] =
                    other[(row, col)].clone();
            }
        }
    }
}

impl<T> Grid2<T> {
    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// `true` if `(row, col)` lies inside the grid (signed inputs so callers
    /// can test neighbour offsets that may go negative).
    #[inline]
    pub fn in_bounds(&self, row: i64, col: i64) -> bool {
        row >= 0 && col >= 0 && (row as usize) < self.rows && (col as usize) < self.cols
    }

    /// The backing row-major data.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The backing row-major data, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Index<(usize, usize)> for Grid2<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        debug_assert!(row < self.rows && col < self.cols);
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Grid2<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        debug_assert!(row < self.rows && col < self.cols);
        &mut self.data[row * self.cols + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_edge_replicates_borders() {
        let grid = Grid2::from_vec(2, 2, vec![1, 2, 3, 4]);
        let padded = grid.pad_edge((4, 4), 1, 1);
        assert_eq!(padded[(0, 0)], 1);
        assert_eq!(padded[(1, 1)], 1);
        assert_eq!(padded[(2, 2)], 4);
        assert_eq!(padded[(3, 3)], 4);
        assert_eq!(padded[(0, 3)], 2);
        assert_eq!(padded[(3, 0)], 3);
    }

    #[test]
    fn pad_const_fills() {
        let grid = Grid2::from_vec(1, 1, vec![7]);
        let padded = grid.pad_const((3, 3), 1, 1, 0);
        assert_eq!(padded[(1, 1)], 7);
        assert_eq!(padded[(0, 0)], 0);
        assert_eq!(padded[(2, 2)], 0);
    }
}
