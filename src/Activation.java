public interface Activation {
    public void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis;
    public void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation;

    public Matrix output();
    public Matrix d_inputs();
}
