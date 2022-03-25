package ann.classifier;

import java.io.Serial;
import java.io.Serializable;

//Sigmoid activation
public class Activation_Sigmoid implements Activation, Serializable {
    private Matrix outputs;
    private Matrix inputs;
    private Matrix d_inputs;

    @Serial
    private static final long serialVersionUID = 6529685098267757614L;

    @Override
    public void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation, InvalidMatrixAxis {
        this.inputs = inputs;
        this.outputs = sigmoid(this.inputs);
    }

    @Override
    public void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixOperation {
        //Derivative - calculates from output of the sigmoid function
        this.d_inputs = d_values.multiply(new Matrix(this.outputs.subtract(1))).multiply(this.outputs);
    }

    @Override
    public double forward(Matrix inputs, Matrix y_true) throws InvalidMatrixOperation, InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        // NOT IN USER
        return 0;
    }

    @Override
    public void backward(Matrix d_values, Matrix y_true) throws InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        // NOT IN USER
    }

    @Override
    public Matrix output() {
        return this.outputs;
    }

    @Override
    public Matrix d_inputs() {
        return this.d_inputs;
    }

    private Matrix sigmoid(Matrix inputs) throws MatrixIndexesOutOfBounds {
        Matrix temp = inputs.multiply(-1).exp().add(1);
        for (int i = 0; i < temp.getRows(); i++) {
            for (int j = 0; j < temp.getColumns(); j++) {
                temp.setValue(i, j, (1 / temp.getValue(i, j)));
            }
        }
        return temp;
    }
}
