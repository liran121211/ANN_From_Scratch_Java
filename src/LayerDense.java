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

//                this.weights = new Dataset().getTest_data(n_inputs,n_neurons);
//                this.biases = new Dataset().getTest_classes(n_neurons);
    }

    /**
     * Backward Pass.
     *
     * @param d_values (Matrix object).
     */
    protected void backward(Matrix d_values) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis, InvalidMatrixOperation {
        this.d_weights = this.inputs.transpose().dot(d_values); //Gradients on parameters
        this.d_biases = d_values.sum(0); //Gradients on parameters
        this.d_inputs = d_values.dot(this.weights.transpose()); //Gradient on values
    }

    /**
     * Forward Pass.
     *
     * @param inputs (Matrix object).
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

    public static void main(String[] args) throws MatrixExceptionHandler {

        //Create dataset
        Matrix X = new Dataset().getSpiral_data();
        Matrix y = new Dataset().getSpiral_classes();

        //Create Dense layer with 2 input features and 3 output values
        LayerDense dense1 = new LayerDense(2, 64);

        //Create ReLU activation (to be used with Dense layer):
        Activation activation1 = new Activation_ReLU();

        //Create second Dense layer with 3 input features (as we take output.
        //of previous layer here) and 3 output values (output values)
        LayerDense dense2 = new LayerDense(64, 3);

        // Create Softmax classifier's combined loss and activation
        Activation_Softmax_Loss_CategoricalCrossEntropy loss_activation = new Activation_Softmax_Loss_CategoricalCrossEntropy();

//        Optimization optimizer = new Optimizer_SGD(1.0,1e-3,0.90);
//        Optimization optimizer = new Optimizer_Adagrad(1e-5);
//        Optimization optimizer = new Optimizer_RMSprop(0.02 ,1e-4 ,0.999);
//        Optimization optimizer = new Optimizer_RMSprop(1e-4 );
        Optimization optimizer = new Optimizer_Adam(0.02,1e-5);

        for (int epoch = 0; epoch < 10001; epoch++) {

            //Perform a forward pass of our training data through this layer
            dense1.forward(X);

            //Perform a forward pass through activation function
            //takes the output of first dense layer here
            activation1.forward(dense1.output);

            //Perform a forward pass through second Dense layer
            //takes outputs of activation function of first layer as inputs
            dense2.forward(activation1.output());

            //Perform a forward pass through the activation/loss function
            //takes the output of second dense layer here and returns loss
            double loss = loss_activation.forward(dense2.output, y);


            //Calculate accuracy from output of activation2 and targets
            //calculate values along first axis
            Matrix predictions = loss_activation.get_outputs().argmax(1);

            if (y.shape() == 2)
                y = new Matrix(y.argmax(1));

            double accuracy = Matrix.bitwiseCompare(predictions.transpose(), y).mean();

            if (epoch % 100 == 0) {
                System.out.println("epoch: " + epoch + " acc: " + accuracy + " loss: " + loss + " lr: " + optimizer.get_current_learning_rate());
            }

            //Backward pass
            loss_activation.backward(loss_activation.get_outputs(), y);
            dense2.backward(loss_activation.get_d_inputs());
            activation1.backward(dense2.d_inputs);
            dense1.backward(activation1.d_inputs());

            optimizer.pre_update_params();
            optimizer.update_params(dense1);
            optimizer.update_params(dense2);
            optimizer.post_update_params();


        }
    }

}
