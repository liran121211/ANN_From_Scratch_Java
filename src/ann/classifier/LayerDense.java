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

    // Layer initialization
    protected LayerDense(int n_inputs, int n_neurons) throws InvalidMatrixDimension, MatrixIndexesOutOfBounds {
        this.weights = Matrix.random(n_inputs, n_neurons).multiply(0.01);
        this.biases = new Matrix(1, n_neurons);

//                this.weights = new ann.classifier.Dataset().getTest_data(n_inputs,n_neurons);
//                this.biases = new ann.classifier.Dataset().getTest_classes(n_neurons);
    }

    /**
     * Backward Pass.
     *
     * @param d_values (ann.classifier.Matrix object).
     */
    protected void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis, InvalidMatrixOperation {
        this.d_weights = this.inputs.transpose().dot(d_values); //Gradients on parameters
        this.d_biases = d_values.sum(0); //Gradients on parameters
        this.d_inputs = d_values.dot(this.weights.transpose()); //Gradient on values
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

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }

    public Matrix getOutput() {
        return output;
    }

    public Matrix getInputs() {
        return inputs;
    }

    public Matrix get_d_weights() {
        return d_weights;
    }

    public Matrix get_d_biases() {
        return d_biases;
    }

    public Matrix get_d_inputs() {
        return d_inputs;
    }

    public Matrix get_weight_momentums() {
        return weight_momentums;
    }

    public Matrix get_bias_momentums() {
        return bias_momentums;
    }

    public Matrix get_weight_cache() {
        return weight_cache;
    }

    public Matrix get_bias_cache() {
        return bias_cache;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    public void setBiases(Matrix biases) {
        this.biases = biases;
    }

    public void set_d_weights(Matrix d_weights) {
        this.d_weights = d_weights;
    }

    public void set_d_biases(Matrix d_biases) {
        this.d_biases = d_biases;
    }

    public void set_weight_momentums(Matrix weight_momentum) {
        this.weight_momentums = weight_momentum;
    }

    public void set_bias_momentums(Matrix bias_momentum) {
        this.bias_momentums = bias_momentum;
    }

    public void set_weight_cache(Matrix weight_cache) {
        this.weight_cache = weight_cache;
    }

    public void set_bias_cache(Matrix bias_cache) {
        this.bias_cache = bias_cache;
    }

    protected static Matrix addBias(Matrix B, Matrix V) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds {
        if (B.getColumns() != V.getColumns())
            throw new MatrixIndexesOutOfBounds(B, V);
        else
            B.add(V);
        return B;

    }
}
