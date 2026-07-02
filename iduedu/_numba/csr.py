import numba as nb
import numpy as np


def _build_csr_matrix_class(name, values_type):
    @nb.experimental.jitclass(
        [
            ("values", values_type),
            ("col_index", nb.uint32[:]),
            ("row_index", nb.uint32[:]),
            ("tot_rows", nb.uint32),
        ]
    )
    class CSRMatrix:
        def __init__(self, values, col_index, row_index):
            self.values = values
            self.col_index = col_index
            self.row_index = row_index
            self.tot_rows = len(row_index) - 1

        def get_cols(self, row):
            start = self.row_index[row]
            end = self.row_index[row + 1]
            return self.col_index[start:end]

        def get_vals(self, row):
            start = self.row_index[row]
            end = self.row_index[row + 1]
            return self.values[start:end]

    CSRMatrix.__name__ = name
    CSRMatrix.__qualname__ = name
    return CSRMatrix


UI32CSRMatrix = _build_csr_matrix_class("UI32CSRMatrix", nb.uint32[:])
BCSRMatrix = _build_csr_matrix_class("BCSRMatrix", nb.boolean[:])


try:
    ui32csr_type = UI32CSRMatrix.class_type.instance_type
    bcsr_type = BCSRMatrix.class_type.instance_type
except Exception:
    ui32csr_type = None
    bcsr_type = None


def sparse_row2numba_matrix(sparse_row_scipy):
    """Convert a SciPy CSR matrix to the uint32 Numba CSR jitclass."""
    values = sparse_row_scipy.data.astype(np.uint32, copy=False)
    col_index = sparse_row_scipy.indices.astype(np.uint32, copy=False)
    row_index = sparse_row_scipy.indptr.astype(np.uint32, copy=False)
    return UI32CSRMatrix(values, col_index, row_index)


def sparse_row2numba_bool_matrix(sparse_row_scipy):
    """Convert a SciPy CSR matrix to the boolean Numba CSR jitclass."""
    values = sparse_row_scipy.data.astype(bool, copy=False)
    col_index = sparse_row_scipy.indices.astype(np.uint32, copy=False)
    row_index = sparse_row_scipy.indptr.astype(np.uint32, copy=False)
    return BCSRMatrix(values, col_index, row_index)


def coo_rows_to_arrays(coo_rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten row-wise COO pairs into row, column and value arrays."""
    nnz = sum(len(row) for row in coo_rows)

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.int32)

    pos = 0
    for row_i, row in enumerate(coo_rows):
        for col_i, distance in row:
            rows[pos] = row_i
            cols[pos] = col_i
            values[pos] = distance
            pos += 1

    return rows, cols, values
