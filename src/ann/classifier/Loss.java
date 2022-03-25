package ann.classifier;

public interface Loss {
    //Calculates the data and regularization losses
    //Given model output and ground truth values

    Matrix forward(Matrix y_pred, Matrix y_true) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds, InvalidMatrixOperation, InvalidMatrixAxis;

    void backward(Matrix d_values, Matrix y_true) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis;

    Matrix d_inputs();

    double calculate(Matrix output);

    double regularization_loss (LayerDense layer) throws MatrixIndexesOutOfBounds;
}
