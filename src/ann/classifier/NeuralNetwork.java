package ann.classifier;

import python.extender.PythonInterpreter;

import java.io.*;
import java.net.DatagramPacket;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NeuralNetwork implements Serializable {
    private int n_inputs;
    private int n_neurons;
    private int n_hidden;
    private int n_outputs;
    private int max_iterations;


    private LayerDense input_layer;
    private LayerDense output_layer;
    private List<LayerDense> hidden_layers;
    private List<LayerDropout> dropout_layers;

    private Optimization optimizer;
    private List<Activation> activations;
    private Activation loss_function;

    private double loss;
    private double accuracy;
    private boolean isLogged;
    private boolean showMetrices;

    private List<String> accuracy_logs;
    private List<String> loss_logs;
    private Matrix predictions;

    @Serial
    private static final long serialVersionUID = 6529685098267757608L;
    private final static String MATRICES_DIR = "bin\\metrices";


    public NeuralNetwork(int n_inputs, int n_neurons, int n_hidden, int n_outputs, String optimizer, String classifier, int max_iterations) throws InvalidMatrixDimension {
        this.n_inputs = n_inputs;
        this.n_neurons = n_neurons;
        this.n_hidden = n_hidden;
        this.n_outputs = n_outputs;
        this.max_iterations = max_iterations;
        this.loss = 0.0;
        this.accuracy = 0.0;
        this.isLogged = false;
        this.showMetrices = false;
        this.accuracy_logs = new ArrayList<>();
        this.loss_logs = new ArrayList<>();

        this.input_layer = (new LayerDense(n_inputs, n_neurons));
        this.output_layer = (new LayerDense(n_neurons, n_outputs));
        this.hidden_layers = new ArrayList<>(n_hidden);
        this.dropout_layers = null;
        for (int i = 0; i < n_hidden; i++)
            this.hidden_layers.add(new LayerDense(n_neurons, n_neurons));

        this.activations = new ArrayList<>(this.n_hidden + 1); // input + hidden + output
        for (int i = 0; i < (this.n_hidden + 1); i++)
            this.activations.add(new Activation_ReLU());

        this.predictions = new Matrix(1, n_outputs);

        //logs headers
        this.accuracy_logs.add("values\n");
        this.loss_logs.add("values\n");

        // set optimizer
        validator(); //check for errors.
        buildOptimizer(optimizer);
        buildClassifier(classifier);
    }

    public NeuralNetwork(int n_inputs, int n_neurons, int n_hidden, int n_outputs, String optimizer, String classifier, int max_iterations, double dropout) throws InvalidMatrixDimension {
        this.n_inputs = n_inputs;
        this.n_neurons = n_neurons;
        this.n_hidden = n_hidden;
        this.n_outputs = n_outputs;
        this.max_iterations = max_iterations;
        this.loss = 0.0;
        this.accuracy = 0.0;
        this.isLogged = false;
        this.showMetrices = false;
        this.accuracy_logs = new ArrayList<>();
        this.loss_logs = new ArrayList<>();

        this.input_layer = (new LayerDense(n_inputs, n_neurons));
        this.output_layer = (new LayerDense(n_neurons, n_outputs));
        this.hidden_layers = new ArrayList<>(n_hidden);
        for (int i = 0; i < n_hidden; i++)
            this.hidden_layers.add(new LayerDense(n_neurons, n_neurons));

        this.dropout_layers = new ArrayList<>(n_hidden + 1); // input layer + (n) hidden layers
        for (int i = 0; i < (n_hidden + 1); i++) {
            this.dropout_layers.add(new LayerDropout(dropout));
        }

        this.activations = new ArrayList<>(this.n_hidden + 1); // input + hidden + output
        for (int i = 0; i < (this.n_hidden + 1); i++)
            this.activations.add(new Activation_ReLU());

        this.predictions = new Matrix(1, n_outputs);

        //logs headers
        this.accuracy_logs.add("values\n");
        this.loss_logs.add("values\n");

        // set optimizer
        validator(); //check for errors.
        buildOptimizer(optimizer);
        buildClassifier(classifier);
    }

    public void fit(Matrix X_train, Matrix y_train) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis, IOException, InvalidMatrixArgument {
        raiseInfo(String.format("Training Started - inputs: %s | hidden_layers: %s | outputs: %s | learning_rate: %s", n_inputs, n_hidden, n_outputs, optimizer.get_learning_rate()));

        for (int epoch = 0; epoch < this.max_iterations; epoch++) {
            //Perform a forward pass of our training data through this layer
            this.input_layer.forward(X_train, true);

            //Perform a forward pass through activation function
            //takes the output of first dense layer here
            this.activations.get(0).forward(this.input_layer.getOutput());

            if (this.dropout_layers != null) // if dropout provided
                this.dropout_layers.get(0).forward(this.activations.get(0).output(), true);

            //Perform a forward pass through second Dense layer
            //takes outputs of activation function of first layer as inputs
            if (this.dropout_layers != null) { // if dropout provided
                for (int i = 0; i < this.n_hidden; i++) {
                    this.hidden_layers.get(i).forward(this.dropout_layers.get(i).getOutput(), true);
                    this.activations.get(i + 1).forward(this.hidden_layers.get(i).getOutput());
                    this.dropout_layers.get(i + 1).forward(this.activations.get(i + 1).output(), true);
                }
            } else { // no dropout provided
                for (int i = 0; i < this.n_hidden; i++) {
                    this.hidden_layers.get(i).forward(this.activations.get(i).output(), true);
                    this.activations.get(i + 1).forward(this.hidden_layers.get(i).getOutput());
                }
            }

            if (this.dropout_layers != null)// if dropout provided
                this.output_layer.forward(this.dropout_layers.get(this.n_hidden).getOutput(), true);
            else
                this.output_layer.forward(this.activations.get(this.n_hidden).output(), true);

            //Perform a forward pass through the activation/loss function
            //takes the output of second dense layer here and returns loss
            double data_loss = this.loss_function.forward(this.output_layer.getOutput(), y_train);

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
            Matrix predictions = calculatePrediction(this.loss_function);

            if (y_train.shape() == 2)
                y_train = new Matrix(y_train.argmax(1));

            // calculate accuracy
            this.accuracy = Matrix.bitwiseCompare(predictions.transpose(), y_train).mean();

            // for analyzing purpose
            double batch_size = ((double) this.max_iterations) / 100;
            if (batch_size < 1)
                batch_size = 1;

            if (epoch % (int) batch_size == 0) {
                if (isLogged)//show logs
                    System.out.printf("Epochs: %d | Accuracy: %.5f | Loss: %.5f | Data Loss: %.10E | Regularization Loss: %.10E%n", epoch, this.accuracy, this.loss, data_loss, regularization_loss);

                //collect logs
                this.accuracy_logs.add(Double.toString(this.accuracy) + '\n');
                this.loss_logs.add(Double.toString(this.loss) + '\n');
            }


            //Backward pass
            int pointer = this.activations.size() - 1; // last object (cell) in activations
            this.loss_function.backward(this.loss_function.output(), y_train);
            this.output_layer.backward(this.loss_function.d_inputs());

            if (this.dropout_layers != null) { // if dropout provided
                this.dropout_layers.get(pointer).backward(this.output_layer.get_d_inputs());
                this.activations.get(pointer).backward(this.dropout_layers.get(pointer).get_d_inputs());
            } else {
                this.activations.get(pointer).backward(output_layer.get_d_inputs());
            }

            if (this.dropout_layers != null) { // if dropout provided
                for (int i = this.n_hidden; i > 0; i--) {
                    this.hidden_layers.get(i - 1).backward(this.dropout_layers.get(pointer--).get_d_inputs());
                    this.activations.get(pointer).backward(this.hidden_layers.get(i - 1).get_d_inputs());
                    this.dropout_layers.get(pointer).backward(this.activations.get(pointer).d_inputs());
                }
            } else {
                for (int i = this.n_hidden; i > 0; i--) {
                    this.hidden_layers.get(i - 1).backward(this.activations.get(pointer--).d_inputs());
                    this.activations.get(pointer).backward(this.hidden_layers.get(i - 1).get_d_inputs());
                }
            }

            if (this.dropout_layers != null) // if dropout provided
                this.input_layer.backward(this.dropout_layers.get(pointer).get_d_inputs());
            else
                this.input_layer.backward(this.activations.get(pointer).d_inputs());

            //Update weights and biases
            this.optimizer.pre_update_params();
            this.optimizer.update_params(this.input_layer);
            for (int i = 0; i < this.n_hidden; i++)
                this.optimizer.update_params(this.hidden_layers.get(i));
            this.optimizer.update_params(this.output_layer);
            this.optimizer.post_update_params();
        }

        if (this.showMetrices) // show animated graphs of metrices.
            this.showMetrices();

        System.out.printf("Training was completed successfully. Accuracy: %.5f | Loss: %.5f%n", this.accuracy, this.loss);
    }

    public void validation(Matrix X_test, Matrix y_test) throws InvalidMatrixOperation, MatrixIndexesOutOfBounds, InvalidMatrixDimension, InvalidMatrixAxis {
        raiseInfo(String.format("Validation Started - inputs: %s | hidden_layers: %s | outputs: %s ", n_inputs, n_hidden, n_outputs));
        //Perform a forward pass of our testing data through this layer
        this.input_layer.forward(X_test, true);


        //Perform a forward pass through activation function
        //takes the output of first dense layer here
        this.activations.get(0).forward(this.input_layer.getOutput());

        //Perform a forward pass through second Dense layer
        //takes outputs of activation function of first layer as inputs
        for (int i = 0; i < this.n_hidden; i++) {
            this.hidden_layers.get(i).forward(this.activations.get(i).output(), true);
            this.activations.get(i + 1).forward(this.hidden_layers.get(i).getOutput());
        }
        this.output_layer.forward(this.activations.get(this.n_hidden).output(), true);

        //Perform a forward pass through the activation/loss function
        //takes the output of second dense layer here and returns loss
        double loss = this.loss_function.forward(this.output_layer.getOutput(), y_test);

        //Calculate accuracy from output of activation2 and targets
        //calculate values along first axis
        Matrix predictions = this.loss_function.output().argmax(1);

        if (y_test.shape() == 2)
            y_test = new Matrix(y_test.argmax(1));

        // calculate accuracy
        this.accuracy = Matrix.bitwiseCompare(predictions.transpose(), y_test).mean();

        System.out.printf("Validation was completed successfully. Accuracy: %.5f | Loss: %.5f%n", this.accuracy, loss);
    }

    public void predict(Matrix X_test) throws InvalidMatrixDimension, InvalidMatrixOperation, MatrixIndexesOutOfBounds, InvalidMatrixAxis {
        raiseInfo(String.format("Testing Started - inputs: %s | hidden_layers: %s | outputs: %s ", n_inputs, n_hidden, n_outputs));

        //Perform a forward pass of our testing data through this layer
        this.input_layer.forward(X_test, true);


        //Perform a forward pass through activation function
        //takes the output of first dense layer here
        this.activations.get(0).forward(this.input_layer.getOutput());

        //Perform a forward pass through second Dense layer
        //takes outputs of activation function of first layer as inputs
        for (int i = 0; i < this.n_hidden; i++) {
            this.hidden_layers.get(i).forward(this.activations.get(i).output(), true);
            this.activations.get(i + 1).forward(this.hidden_layers.get(i).getOutput());
        }
        this.output_layer.forward(this.activations.get(this.n_hidden).output(), true);


        //Perform a forward pass through the activation/loss function
        //takes the output of second dense layer here and returns loss
        this.loss_function.forward(this.output_layer.getOutput());

        //Calculate predictions
        this.predictions = new Matrix(this.loss_function.output().argmax(1));
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

    private void buildClassifier(String name) {
        if (name.toLowerCase().compareTo("binary") == 0)
            this.loss_function = new Activation_Sigmoid_Loss_BinaryCrossEntropy();
        else if (name.toLowerCase().compareTo("categorical") == 0)
            this.loss_function = new Activation_Softmax_Loss_CategoricalCrossEntropy();
        else {
            raiseWarning(String.format("[%s] classifier does not exist, setting to [categorical] by default...", name));
            this.loss_function = new Activation_Softmax_Loss_CategoricalCrossEntropy();
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

    public Matrix get_predictions() {
        return this.predictions;
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

    public void setMetrices(boolean status) {
        this.showMetrices = status;
    }

    public void set_max_iterations(int max_iterations) {
        this.max_iterations = max_iterations;
    }

    public void showMetrices() throws IOException {
        FileWriter accuracy_writer = new FileWriter(NeuralNetwork.DirectoryAssurance(MATRICES_DIR, "accuracy_logs.csv"));
        String accuracy_data = String.join(",", this.accuracy_logs);
        accuracy_writer.write(accuracy_data);
        accuracy_writer.close();

        FileWriter loss_writer = new FileWriter(NeuralNetwork.DirectoryAssurance(MATRICES_DIR, "loss_logs.csv"));
        String loss_data = String.join(",", this.loss_logs);
        loss_writer.write(loss_data);
        loss_writer.close();

        // execute python script
        System.out.println("Preparing data for visualization...");
        PythonInterpreter.exec("python\\metrices_animation.py");
    }

    private Matrix calculatePrediction(Activation loss_function) throws InvalidMatrixAxis, MatrixIndexesOutOfBounds, InvalidMatrixDimension {
        if (loss_function instanceof Activation_Softmax_Loss_CategoricalCrossEntropy)
            return this.loss_function.output().argmax(1);

        if (loss_function instanceof Activation_Sigmoid_Loss_BinaryCrossEntropy) {
            for (int i = 0; i < this.loss_function.output().getRows(); i++) {
                for (int j = 0; j < this.loss_function.output().getColumns(); j++) {
                    if (this.loss_function.output().getValue(i, j) > 0.5)
                        this.loss_function.output().setValue(i, j, 1);
                    else
                        this.loss_function.output().setValue(i, j, 0);
                }
            }
            return this.loss_function.output();
        } else
            throw new Error("loss function does not exists");
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

    private static File DirectoryAssurance(String directory, String filename) {
        File dir = new File(directory);
        if (!dir.exists())
            dir.mkdirs();

        return new File(directory + "/" + filename);
    }

    public static void SaveObject(NeuralNetwork obj) throws IOException {
        String fileName = "Doodle_NN.txt";
        FileOutputStream fos = new FileOutputStream(fileName);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(obj);
        oos.close();
    }

    public static NeuralNetwork LoadObject(String fname) throws IOException, ClassNotFoundException {
        FileInputStream fin = new FileInputStream(fname);
        ObjectInputStream ois = new ObjectInputStream(fin);
        NeuralNetwork neural_network = (NeuralNetwork) ois.readObject();
        ois.close();
        return neural_network;
    }
}

