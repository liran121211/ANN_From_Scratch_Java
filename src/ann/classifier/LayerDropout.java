package ann.classifier;

import java.io.Serial;
import java.io.Serializable;

public class LayerDropout implements Serializable {
    private final double rate;
    private Matrix outputs;
    private Matrix d_inputs;
    private Matrix binary_mask;

    @Serial
    private static final long serialVersionUID = 6529685098267757613L;

    protected LayerDropout(double rate) {
        // Store rate, we invert it as for example for dropout
        // of 0.1 we need success rate of 0.9
        this.rate = 1 - rate;
    }

    protected void forward(Matrix inputs, boolean training) throws MatrixIndexesOutOfBounds, InvalidMatrixOperation, InvalidMatrixDimension, InvalidMatrixArgument {
        Matrix input = new Matrix(inputs);

        //Generate and save scaled mask
        this.binary_mask = Matrix.binomial(1, this.rate, inputs.getRows()).divide(this.rate);

        //Apply mask to output values
        this.outputs = input.multiply(this.binary_mask);
    }

    protected void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis, InvalidMatrixOperation {
        //Gradient on values
        this.d_inputs = d_values.multiply(this.binary_mask);
    }

    protected Matrix getOutput() {
        return this.outputs;
    }

    protected Matrix get_d_inputs() {
        return d_inputs;
    }
}
