public class Activation_SoftMax implements Activation {
    private Matrix outputs;
    private Matrix inputs;
    private Matrix d_inputs;

    @Override
    public void forward(Matrix inputs) throws InvalidMatrixDimension, InvalidMatrixOperation, MatrixIndexesOutOfBounds, InvalidMatrixAxis {
        this.inputs = inputs; // Remember input values

        //Get unnormalized probabilities
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        Matrix np_max = new Matrix(inputs.max(1));
        Matrix subtract = new Matrix(inputs.subtract(np_max));
        Matrix exp_values = new Matrix(subtract.exp());

        //Normalize them for each sample
        this.outputs = exp_values.divide(exp_values.sum(1));
    }

    @Override
    public void backward(Matrix d_values) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds, InvalidMatrixOperation {
        this.d_inputs = new Matrix(d_values.getRows(), d_values.getColumns()); // Create zeros array (uninitialized)

        for (int i = 0; i < d_values.getRows(); i++) { // like enumerate(zip(this.output,d_values))
            Matrix single_output = this.outputs.getRow(i).reshape(-1, 1); // Flatten output array
            Matrix jacobian_matrix = Matrix.diagflat(single_output).subtract(single_output.dot(single_output.transpose())); // Calculate Jacobian matrix of the output

            // Calculate sample-wise gradient
            // and add it to the array of sample gradients
            Matrix vector = jacobian_matrix.dot(d_values.getRow(i).transpose());
            for (int j = 0; j < d_inputs.getRows(); j++) {
                d_inputs.setValue(j, i, vector.getValue(j, 0));
            }
        }

    }

    public Matrix d_inputs() {
        return d_inputs;
    }

    @Override
    public Matrix output() {
        return outputs;
    }

    public Matrix inputs() {
        return inputs;
    }
}
