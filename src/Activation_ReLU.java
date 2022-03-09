public class Activation_ReLU implements Activation {
    private Matrix outputs;
    private Matrix inputs;
    private Matrix d_inputs;

    @Override
    public void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        this.inputs = inputs;
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        for (int i = 0; i < inputs.getRows(); i++) {
            for (int j = 0; j < inputs.getColumns(); j++)
                this.outputs.setValue(i, j, Math.max(0.0, inputs.getValue(i, j)));
        }
    }

    @Override
    public void backward(Matrix d_values) throws MatrixIndexesOutOfBounds {
        this.d_inputs = new Matrix(d_values);
        for (int i = 0; i < d_inputs.getRows(); i++) {
            for (int j = 0; j < d_inputs.getColumns(); j++) {
                if (this.inputs.getValue(i, j) <= 0)
                    this.d_inputs.setValue(i, j, 0.0);
            }
        }
    }

    @Override
    public Matrix output() {
        return outputs;
    }

    @Override
    public Matrix d_inputs() {
        return d_inputs;
    }
}
