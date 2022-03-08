import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.Arrays;

public class LayerDense {
    private Matrix weights;
    private Matrix biases;
    private Matrix output;
    private Matrix inputs;
    private Matrix d_weights;
    private Matrix d_biases;
    private Matrix d_inputs;

    protected LayerDense(int n_inputs, int n_neurons) {
        this.weights = Matrix.random(n_inputs, n_neurons).multiply(0.01);
        this.biases = new Matrix(1, n_neurons);
    }

    protected void backward(Matrix d_values) {
        this.d_weights = this.inputs.transpose().dot(d_values); //Gradients on parameters
        this.d_biases = d_values.sum(0); //Gradients on parameters
        this.d_inputs = d_values.dot(this.weights.transpose()); //Gradient on values

    }

    /**
     * Calculate output values from inputs, weights and biases
     *
     * @param inputs (Matrix object).
     */
    protected void forward(Matrix inputs) {
        this.inputs = new Matrix(inputs);
        this.output = addBias(inputs.dot(this.weights), this.biases);
    }


    protected static Matrix addBias(Matrix B, Matrix V) throws IndexOutOfBoundsException {
        if (B.getColumns() != V.getColumns())
            throw new IndexOutOfBoundsException(String.format("Matrices has different dimensions (%s,%s) By (%s,%s)", B.getRows(), B.getColumns(), V.getRows(), V.getColumns()));
        else
            B.add(V);
        return B;

    }

    public static void main(String[] args){

        //Create dataset
        Matrix X = new Dataset().getSpiral_data();
        Matrix y = new Dataset().getSpiral_classes();

        //Create Dense layer with 2 input features and 3 output values
        LayerDense dense1 = new LayerDense(2, 3);

        //Create ReLU activation (to be used with Dense layer):
        Activation activation1 = new Activation_ReLU();

        //Create second Dense layer with 3 input features (as we take output.
        //of previous layer here) and 3 output values (output values)
        LayerDense dense2 = new LayerDense(3, 3);

        // Create Softmax classifier's combined loss and activation
        Activation_Softmax_Loss_CategoricalCrossEntropy loss_activation = new Activation_Softmax_Loss_CategoricalCrossEntropy();

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

        //Let's see output of the first few samples:
        System.out.println(loss_activation.get_outputs());

        //Print loss value
        System.out.println("loss: " + loss);

        //Calculate accuracy from output of activation2 and targets
        //calculate values along first axis
        Matrix predictions = loss_activation.get_outputs().argmax(1);

        if (y.shape() == 2)
            y = new Matrix(y.argmax(1));
        double accuracy = Matrix.bitwiseCompare(predictions.transpose(), y).mean();

        //Print accuracy
        System.out.println("accuracy: " + accuracy);

        //Backward pass
        loss_activation.backward(loss_activation.get_outputs(), y);
        dense2.backward(loss_activation.get_d_inputs());
        activation1.backward(dense2.d_inputs);
        dense1.backward(activation1.d_inputs());

        //Print gradients
        System.out.println(dense1.d_weights);
        System.out.println(dense1.d_biases);
        System.out.println(dense2.d_weights);
        System.out.println(dense2.d_biases);
    }

}
