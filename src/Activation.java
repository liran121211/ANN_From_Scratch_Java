public interface Activation {
    void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis;

    void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation;

    Matrix output();

    Matrix d_inputs();
}
