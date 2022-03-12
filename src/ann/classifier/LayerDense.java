package ann.classifier;

public class LayerDense {
    private Matrix weights;
    private Matrix biases;

    private Matrix output;
    private Matrix inputs;

    private Matrix d_weights;
    private Matrix d_biases;
    private Matrix d_inputs;

    private Matrix weight_momentums;
    private Matrix bias_momentums;

    private Matrix weight_cache;
    private Matrix bias_cache;

    private double weight_regularizer_l1; // for penalty
    private double weight_regularizer_l2; // for penalty
    private double bias_regularizer_l1; // for penalty
    private double bias_regularizer_l2; // for penalty

    // Layer initialization
    protected LayerDense(int n_inputs, int n_neurons) throws InvalidMatrixDimension {
        this.weights = Matrix.random(n_inputs, n_neurons).multiply(0.01);
        this.biases = new Matrix(1, n_neurons);
        this.weight_regularizer_l1 = 0.0;
        this.weight_regularizer_l2 = 0.0;
        this.bias_regularizer_l1 = 0.0;
        this.bias_regularizer_l2 = 0.0;
    }

    /**
     * Backward Pass.
     *
     * @param d_values (ann.classifier.Matrix object).
     */
    protected void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis, InvalidMatrixOperation {
        //Gradients on parameters
        this.d_weights = this.inputs.transpose().dot(d_values);
        this.d_biases = d_values.sum(0);

        //Gradients on regularization
        //L1 on weights
        if (this.weight_regularizer_l1 > 0) {
            Matrix d_l1 = Matrix.ones_like(this.weights);
            for (int i = 0; i < this.weights.getRows(); i++) {
                for (int j = 0; j < this.weights.getColumns(); j++)
                    if (this.weights.getValue(i, j) < 0)
                        d_l1.setValue(i, j, -1);
            }
            this.d_weights = this.d_weights.add(d_l1.multiply(this.weight_regularizer_l1));
        }

        //L2 on weights
        if (this.weight_regularizer_l2 > 0)
            this.d_weights = this.d_weights.add(new Matrix(this.weights).multiply(2 * this.weight_regularizer_l2));

        //L1 on biases
        if (this.bias_regularizer_l1 > 0) {
            Matrix d_l1 = Matrix.ones_like(this.biases);
            for (int i = 0; i < this.biases.getRows(); i++) {
                for (int j = 0; j < this.biases.getColumns(); j++)
                    if (this.biases.getValue(i, j) < 0)
                        d_l1.setValue(i, j, -1);
            }
            this.d_biases = this.d_biases.add(d_l1.multiply(this.bias_regularizer_l1));
        }

        //L2 on biases
        if (this.bias_regularizer_l2 > 0)
            this.d_biases = this.d_biases.add(new Matrix(this.biases).multiply(2 * this.bias_regularizer_l2));

        //Gradient on values
        this.d_inputs = d_values.dot(this.weights.transpose());

    }

    /**
     * Forward Pass.
     *
     * @param inputs (ann.classifier.Matrix object).
     */
    protected void forward(Matrix inputs) throws MatrixIndexesOutOfBounds, InvalidMatrixOperation, InvalidMatrixDimension {
        this.inputs = inputs; //Remember input values
        this.output = addBias(inputs.dot(this.weights), this.biases); //Calculate output values from inputs, weights and biases
    }

    protected Matrix getWeights() {
        return weights;
    }

    protected Matrix getBiases() {
        return biases;
    }

    protected Matrix getOutput() {
        return output;
    }

    protected Matrix getInputs() {
        return inputs;
    }

    protected Matrix get_d_weights() {
        return d_weights;
    }

    protected Matrix get_d_biases() {
        return d_biases;
    }

    protected Matrix get_d_inputs() {
        return d_inputs;
    }

    protected Matrix get_weight_momentums() {
        return weight_momentums;
    }

    protected Matrix get_bias_momentums() {
        return bias_momentums;
    }

    protected Matrix get_weight_cache() {
        return weight_cache;
    }

    protected Matrix get_bias_cache() {
        return bias_cache;
    }

    protected double get_weight_regularizer_l1() {
        return weight_regularizer_l1;
    }

    protected double get_weight_regularizer_l2() {
        return weight_regularizer_l2;
    }

    protected double get_bias_regularizer_l1() {
        return bias_regularizer_l1;
    }

    protected double get_bias_regularizer_l2() {
        return bias_regularizer_l2;
    }

    protected void setWeights(Matrix weights) {
        this.weights = weights;
    }

    protected void setBiases(Matrix biases) {
        this.biases = biases;
    }

    protected void set_d_weights(Matrix d_weights) {
        this.d_weights = d_weights;
    }

    protected void set_d_biases(Matrix d_biases) {
        this.d_biases = d_biases;
    }

    protected void set_weight_momentums(Matrix weight_momentum) {
        this.weight_momentums = weight_momentum;
    }

    protected void set_bias_momentums(Matrix bias_momentum) {
        this.bias_momentums = bias_momentum;
    }

    protected void set_weight_cache(Matrix weight_cache) {
        this.weight_cache = weight_cache;
    }

    protected void set_bias_cache(Matrix bias_cache) {
        this.bias_cache = bias_cache;
    }

    protected void set_weight_regularizer_l1(double weight_regularizer_l1) {
        this.weight_regularizer_l1 = weight_regularizer_l1;
    }

    protected void set_weight_regularizer_l2(double weight_regularizer_l2) {
        this.weight_regularizer_l2 = weight_regularizer_l2;
    }

    protected void set_bias_regularizer_l1(double bias_regularizer_l1) {
        this.bias_regularizer_l1 = bias_regularizer_l1;
    }

    protected void set_bias_regularizer_l2(double bias_regularizer_l2) {
        this.bias_regularizer_l2 = bias_regularizer_l2;
    }

    protected static Matrix addBias(Matrix B, Matrix V) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds {
        if (B.getColumns() != V.getColumns())
            throw new MatrixIndexesOutOfBounds(B, V);
        else
            B.add(V);
        return B;

    }
}
