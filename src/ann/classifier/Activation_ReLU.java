package ann.classifier;

public class Activation_ReLU implements Activation {
    private Matrix outputs;
    private Matrix inputs;
    private Matrix d_inputs;

    @Override
    // Forward pass
    public void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        this.inputs = inputs; // Remember input values

        //Calculate output values from inputs
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        for (int i = 0; i < inputs.getRows(); i++) {
            for (int j = 0; j < inputs.getColumns(); j++)
                this.outputs.setValue(i, j, Math.max(0.0, inputs.getValue(i, j)));
        }
    }

    @Override
    // Backward pass
    public void backward(Matrix d_values) throws MatrixIndexesOutOfBounds {
        // Since we need to modify original variable,
        // Need make a copy of values first
        this.d_inputs = new Matrix(d_values);

        // Zero gradient where input values were negative
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

    @Override
    public double forward(Matrix inputs, Matrix y_true)  {
        //NOT IN USE
        return 0.0;
    }

    @Override
    public void backward(Matrix d_values, Matrix y_true)  {
        //NOT IN USE
    }
}
