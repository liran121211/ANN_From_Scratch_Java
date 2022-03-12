package ann.classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NeuralNetwork {
    private int n_inputs;
    private int n_neurons;
    private int n_hidden;
    private int n_outputs;
    private int max_iterations;


    private LayerDense input_layer;
    private LayerDense output_layer;
    private List<LayerDense> hidden_layers;

    private Optimization optimizer;
    private List<Activation> activations;
    private Activation loss_activation;

    private double loss;
    private double accuracy;
    private boolean isLogged;


    public NeuralNetwork(int n_inputs, int n_neurons, int n_hidden, int n_outputs, String optimizer, int max_iterations) throws MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        this.n_inputs = n_inputs;
        this.n_neurons = n_neurons;
        this.n_hidden = n_hidden;
        this.n_outputs = n_outputs;
        this.max_iterations = max_iterations;
        this.loss = 0.0;
        this.accuracy = 0.0;
        this.isLogged = false;

        this.input_layer = (new LayerDense(n_inputs, n_neurons));
        this.output_layer = (new LayerDense(n_neurons, n_outputs));
        this.hidden_layers = new ArrayList<>(n_hidden);
        for (int i = 0; i < n_hidden; i++)
            this.hidden_layers.add(new LayerDense(n_neurons, n_neurons));

        this.activations = new ArrayList<>(this.n_hidden + 1); // input + hidden + output
        for (int i = 0; i < (this.n_hidden + 1); i++)
            this.activations.add(new Activation_ReLU());

        // set optimizer
        validator(); //check for errors.
        buildOptimizer(optimizer);
    }

    public void fit(Matrix X_train, Matrix y_train) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis {
        raiseInfo(String.format("Training Started - inputs: %s | hidden_layers: %s | outputs: %s | learning_rate: %s", n_inputs, n_hidden, n_outputs, optimizer.get_learning_rate()));
        this.loss_activation = new Activation_Softmax_Loss_CategoricalCrossEntropy();

        for (int epoch = 0; epoch < this.max_iterations; epoch++) {
            //Perform a forward pass of our training data through this layer
            this.input_layer.forward(X_train);

            //Perform a forward pass through activation function
            //takes the output of first dense layer here
            this.activations.get(0).forward(this.input_layer.getOutput());


            //Perform a forward pass through second Dense layer
            //takes outputs of activation function of first layer as inputs
            for (int i = 0; i < this.n_hidden; i++) {
                this.hidden_layers.get(i).forward(this.activations.get(i).output());
                this.activations.get(i + 1).forward(this.hidden_layers.get(i).getOutput());
            }
            this.output_layer.forward(this.activations.get(this.n_hidden).output());

            //Perform a forward pass through the activation/loss function
            //takes the output of second dense layer here and returns loss
            double data_loss = this.loss_activation.forward(this.output_layer.getOutput(), y_train);

            // Calculate regularization penalty
            double regularization_loss = 0.0;
            regularization_loss += Activation_Softmax_Loss_CategoricalCrossEntropy.regularization_loss(input_layer);
            for (int i = 0; i < this.n_hidden; i++)
                regularization_loss += Activation_Softmax_Loss_CategoricalCrossEntropy.regularization_loss(this.hidden_layers.get(i));
            regularization_loss += Activation_Softmax_Loss_CategoricalCrossEntropy.regularization_loss(output_layer);

            //Calculate overall loss
            this.loss = data_loss + regularization_loss;

            //Calculate accuracy from output of activation2 and targets
            //calculate values along first axis
            Matrix predictions = this.loss_activation.output().argmax(1);

            if (y_train.shape() == 2)
                y_train = new Matrix(y_train.argmax(1));

            // calculate accuracy
            this.accuracy = Matrix.bitwiseCompare(predictions.transpose(), y_train).mean();

            //show logs
            if (isLogged)
                if (epoch % 50 == 0)
                    System.out.printf("Epochs: %d | Accuracy: %.5f | Loss: %.5f | Data Loss: %.5f | Regularization Loss: %.5f%n", epoch, this.accuracy, this.loss, data_loss, regularization_loss);


            //Backward pass
            int pointer = this.activations.size() - 1; // last object (cell) in activations
            this.loss_activation.backward(this.loss_activation.output(), y_train);

            this.output_layer.backward(this.loss_activation.d_inputs());
            this.activations.get(pointer).backward(output_layer.get_d_inputs());
            for (int i = this.n_hidden; i > 0; i--) {
                this.hidden_layers.get(i - 1).backward(this.activations.get(pointer--).d_inputs());
                this.activations.get(pointer).backward(this.hidden_layers.get(i - 1).get_d_inputs());
            }
            this.input_layer.backward(this.activations.get(pointer).d_inputs());

            //Update weights and biases
            this.optimizer.pre_update_params();
            this.optimizer.update_params(this.input_layer);
            for (int i = 0; i < this.n_hidden; i++)
                this.optimizer.update_params(this.hidden_layers.get(i));
            this.optimizer.update_params(this.output_layer);
            this.optimizer.post_update_params();
        }
        System.out.printf("Training was completed successfully. Accuracy: %.5f | Loss: %.5f%n", this.accuracy, this.loss);
    }

    public void predict(Matrix X_test, Matrix y_test) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis {
        raiseInfo(String.format("Testing Started - inputs: %s | hidden_layers: %s | outputs: %s | learning_rate: %s", n_inputs, n_hidden, n_outputs, optimizer.get_learning_rate()));
        //Perform a forward pass of our testing data through this layer
        this.input_layer.forward(X_test);


        //Perform a forward pass through activation function
        //takes the output of first dense layer here
        this.activations.get(0).forward(this.input_layer.getOutput());

        //Perform a forward pass through second Dense layer
        //takes outputs of activation function of first layer as inputs
        for (int i = 0; i < this.n_hidden; i++) {
            this.hidden_layers.get(i).forward(this.activations.get(i).output());
            this.activations.get(i + 1).forward(this.hidden_layers.get(i).getOutput());
        }
        this.output_layer.forward(this.activations.get(this.n_hidden).output());

        //Perform a forward pass through the activation/loss function
        //takes the output of second dense layer here and returns loss
        double loss = this.loss_activation.forward(this.output_layer.getOutput(), y_test);

        //Calculate accuracy from output of activation2 and targets
        //calculate values along first axis
        Matrix predictions = this.loss_activation.output().argmax(1);

        if (y_test.shape() == 2)
            y_test = new Matrix(y_test.argmax(1));

        // calculate accuracy
        this.accuracy = Matrix.bitwiseCompare(predictions.transpose(), y_test).mean();

        System.out.printf("Testing was completed successfully. Accuracy: %.5f | Loss: %.5f%n", this.accuracy, this.loss);


    }

    private void buildOptimizer(String name) {
        if (name.toLowerCase().compareTo("sgd") == 0)
            this.optimizer = new Optimizer_SGD();
        else if (name.toLowerCase().compareTo("rms_prop") == 0)
            this.optimizer = new Optimizer_RMSprop();
        else if (name.toLowerCase().compareTo("adagrad") == 0)
            this.optimizer = new Optimizer_Adagrad();
        else if (name.toLowerCase().compareTo("adam") == 0)
            this.optimizer = new Optimizer_Adam();
        else {
            raiseWarning(String.format("[%s] optimizer does not exist, setting to [adam] by default...", name));
            this.optimizer = new Optimizer_Adam();
        }
    }

    private void validator() {
        if (this.n_inputs <= 0) {
            raiseFatalError("cannot initialize network with zero or less inputs!");
            System.exit(-1);
        }
        if (this.n_hidden <= 0) {
            raiseFatalError("cannot initialize network with zero or less hidden layers!");
            System.exit(-1);
        }
        if (this.n_outputs <= 0) {
            raiseFatalError("cannot initialize network with zero or less outputs!");
            System.exit(-1);
        }
        if (this.max_iterations <= 0) {
            raiseFatalError("cannot initialize network with zero or epochs!");
            System.exit(-1);
        }
    }

    /**
     * Set L1 regularization to penalty the network.
     *
     * @param weight_regularizer linear penalty (double).
     * @param bias_regularizer   non-linear penalty (double).
     */
    public void set_l1_regularizer(double weight_regularizer, double bias_regularizer) {
        this.input_layer.set_weight_regularizer_l1(weight_regularizer);
        this.input_layer.set_bias_regularizer_l1(bias_regularizer);
    }

    /**
     * Set L2 regularization to penalty the network.
     *
     * @param weight_regularizer linear penalty (double).
     * @param bias_regularizer   non-linear penalty (double).
     */
    public void set_l2_regularizer(double weight_regularizer, double bias_regularizer) {
        this.input_layer.set_weight_regularizer_l2(weight_regularizer);
        this.input_layer.set_bias_regularizer_l2(bias_regularizer);
    }

    public void setLogs(boolean status) {
        this.isLogged = status;
    }

    private static void raiseWarning(String msg) {
        Logger logger = Logger.getLogger(NeuralNetwork.class.getName());
        logger.setLevel(Level.WARNING);
        logger.warning(msg);
    }

    private static void raiseFatalError(String msg) {
        Logger logger = Logger.getLogger(NeuralNetwork.class.getName());
        logger.setLevel(Level.SEVERE);
        logger.severe(msg);
    }

    private static void raiseInfo(String msg) {
        final String ANSI_RESET = "\u001B[0m";
        final String ANSI_BLUE = "\u001B[34m";
        System.out.println(ANSI_BLUE + "INFO: " + msg + ANSI_RESET);
    }
}

