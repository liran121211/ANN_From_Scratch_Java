
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://i.ibb.co/PFKPvqS/ICLH-Diagram-Batch-01-03-Deep-Neural-Network-WHITEBG.webp" alt="Logo" width="672" height="378">
  </a>

  <h3 align="center">Neural Network From Scratch in Java</h3>

  <h4 align="center">
    Train Your Own Model With Simple Steps...
    <h4 />
  
  
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#limitations">Limitations</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>
  
  <!-- ABOUT THE PROJECT -->
## About The Project
  Inspired By the book (Harrison Kinsley, Daniel Kukieła - Neural Networks from Scratch in Python) I decided to to implement every single line in pure java for educational purposes.
  This Neural Network is very flexible, Allows to choose as much inputs, hidden layers, outputs as possible.
  This project core can be used for every idea you can come up with, like image recognition, datasets classification, etc...
  
### Built With
 - Pure Java
  **No need for extra libraries to run this**
  
<!-- GETTING STARTED -->
## Getting Started
  import the library, and create a new object of [NeuralNetwork ann = new NeuralNetwork(n_inputs, n_neurons, n_hidden_layers, n_outputs, OPTIMIZER, ITERATIONS);]
  use [fit(Matrix X_train, Matrix y_train)] function and send train data and train classifications.
  use [predict(Matrix X_test, Matrix y_test)] function and send test data and test classifications / use can send your test data without classifications and get predictions for new data.
 
  
<!-- ROADMAP -->
## Features
  4 Optimizers:
    - SGD: Stochastic Gradient Descent
    - Adagrad: Adaptive Gradient
    - RMSprop: Root Mean Square Propagation
    - Adam: Adaptive Momentum
  Activation Functions: SoftMax, ReLU
  Classification Type: CategorialCrossEntropy
  
<!-- LICENSE -->
## License

Distributed under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) License. 



<!-- CONTACT -->
## Contact

Liran Smadja - [@LinkedIn](https://www.linkedin.com/in/liran-smadja/) - liransm@ac.sce.ac.il

Other Projects: [See Now!](https://github.com/liran121211)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
  Many thanks to (Harrison Kinsley, Daniel Kukieła) with their great help from Neural Networks from Scratch - https://nnfs.io/
  
  
