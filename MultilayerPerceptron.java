import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * Implements a A-B-C-D network. Trains network using backpropagation.
 * 
 * The default configuration file is input.txt. 
 * The configuration file can be overridden by passing another file path to the command line.
 *
 * If no output file name is specified in the configuration file, the default will be used to save the last known weights.
 * The default output file is output.txt.
 * 
 * 
 * Format of Input File:
 * R           //R for running, T for training
 * P           //R for randomized initial weights, P for preloaded initial weights
 * 4           //number of layers
 * 2 100 20 3  //number of units in each layer 
 * 4           //number of training sets
 * 1 1 1 1 0   //training sets 1-4:
 * 1 0 0 1 1
 * 0 1 0 1 1 
 * 0 0 0 0 0
 * 500         //maximum number of iterations 
 * 0.3         //minimum error
 * 3           //learning rate (lambda)  
 * -1 1        //minimum weight value and maximum weight value
 * 6           //number of iterations to pass before weights are saved
 * out.txt     //the output file that weights will be saved to
 * read.txt    //the file that the weights will be read from (if initial weights are preloaded)
 * 
 * 
 * Table of Contents:
 * public double activationDerivative(double x) 
 * public double activationFunction(double x)
 * public void allocateMemory()
 * public void backpropagation(int trainingSet)
 * public double calculateError(double expected, double calculated)
 * public void calculateNode()
 * public void echo()
 * public void forwardPass(int trainingSet)
 * public double getTotalError()
 * public void printWeights()
 * public void readData(String file)
 * public void runTrainingSet()
 * public void saveControl()
 * public void saveWeights(int iteration)
 * public void setOutputFileName (String filename)
 * public void train()
 *
 * @author Annmaria Antony
 * @version April 21, 2022
 */
public class Perceptron3 
{

   private int numLayers;
   private String mode;
   private int[] numNodes;
   private int numInputs;
   private int numHiddenLayers1;
   private int numHiddenLayers2;
   private int numOutputs;
   private static int numTrainingSets;
   private double[][] trainingSets;
   private int maxNumberIterations;
   private double minError;
   private double lambda;
   private double minWeight;
   private double maxWeight;
   private double[][][] weights;
   private int deepestLayer;
   private double[][] nodeCalculations;
   private double[][] originalNodeCalculations;
   private double[] omega;
   private String preLoad;
   private int writeFrequency;
   private String outputFile;
   private String readFile;
   double[] psiI;
   double[] psiJ;
   double psiK;
   double[] thetai;
   double[] thetaj;
   double[] thetak;  
   double omegaJ;
   double omegaK;
   double[] hiddens1;
   double[] hiddens2;
   double[] outputs;
   
   
   /**
    * Executes the methods of the perceptron.
    * 
    * @param args arguments from the command line
    */
   public static void main(String[] args) 
   {
      
      String inputFile = "input.txt";
      String outputFile = "output.txt";
      Perceptron3 obj = new Perceptron3();

      if (args.length > 0)
      {
         inputFile = args[0]; 
         if (args.length > 1)
         {
             outputFile = args[1];
         }
      }
  
      obj.setOutputFileName(outputFile);
      obj.readData(inputFile);
      obj.echo();
      System.out.println();

      if (obj.mode.equals("R")) 
      {
         obj.runTrainingSet();
      }
      else 
      {
         obj.train();
      }
      
   } // public static void main(String[] args) 
   
   
   /**
    * Allocates array space in memory.
    */
   public void allocateMemory()
   {
       
       if (mode.equals("T"))
       {
          omega = new double[deepestLayer];
          thetai = new double[numOutputs];
          thetaj = new double[numHiddenLayers2];
          thetak = new double[numHiddenLayers1];
          psiI = new double[numOutputs];
          psiJ = new double[numHiddenLayers2];
       }
       
       numNodes = new int[numLayers]; 
       hiddens1 = new double[numHiddenLayers1];
       hiddens2 = new double[numHiddenLayers2];
       outputs = new double[numOutputs];
       trainingSets = new double[numTrainingSets][numInputs + numOutputs];
       weights = new double[numLayers][deepestLayer][deepestLayer];
       nodeCalculations = new double[numLayers][deepestLayer];
       originalNodeCalculations = new double[numLayers][deepestLayer];
       
   } // public void allocateMemory()

   
   /**
    * Echoes the configuration of the perceptron. Prints to the terminal.
    */
   public void echo() 
   {
      
      System.out.println("CONFIGURATION:");
      System.out.println();
      System.out.println("number of layers: " + numLayers);
      System.out.print("number of nodes in each layer: ");

      for (int a = 0; a < numLayers; a++) 
      {
         System.out.print(numNodes[a] + " ");
      } 

      System.out.println();
      System.out.println("number of inputs: " + numInputs);
      System.out.println("number of outputs: " + numOutputs);
      System.out.println("number of training sets: " + numTrainingSets);
      System.out.println();
      System.out.println("training sets: ");

      for (int b = 0; b < numTrainingSets; b++) 
      {
         
         for (int c = 0; c < numInputs + numOutputs; c++) 
         {
            System.out.print(trainingSets[b][c] + " ");
         } 
         
         System.out.println();
         
      } // for (int b = 0; b < numTrainingSets; b++)

      System.out.println();
      System.out.println("maximum number of iterations: " + maxNumberIterations);
      System.out.println("minimum error threshold: " + minError);
      System.out.println("learning factor (lambda): " + lambda);
      System.out.println("minimum weight: " + minWeight);
      System.out.println("maximum weight: " + maxWeight);
      System.out.println();
      System.out.println("initial weights: ");
      
   } // public void echo()

   
   /**
    * Prints the configuration of the perceptron to the output file.
    */
   public void saveControl() 
   {
      
       try 
       {
           PrintWriter pw = new PrintWriter(new File(outputFile));
           pw.println("CONFIGURATION:");
           pw.println();
           pw.println("number of layers: " + numLayers);
           pw.print("number of nodes in each layer: ");

           for (int a = 0; a < numLayers; a++) 
           {
              pw.print(numNodes[a] + " ");
           } 

           pw.println();
           pw.println("number of inputs: " + numInputs);
           pw.println("number of outputs: " + numOutputs);
           pw.println("number of training sets: " + numTrainingSets);
           pw.println();
           pw.println("training sets: ");

           for (int b = 0; b < numTrainingSets; b++) 
           {
              
              for (int c = 0; c < numInputs + numOutputs; c++) 
              {
                  pw.print(trainingSets[b][c] + " ");
              } 
              
              pw.println();
              
           } // for (int b = 0; b < numTrainingSets; b++)

           pw.println();
           pw.println("maximum number of iterations: " + maxNumberIterations);
           pw.println("minimum error threshold: " + minError);
           pw.println("learning factor (lambda): " + lambda);
           pw.println("minimum weight: " + minWeight);
           pw.println("maximum weight: " + maxWeight);
      
           pw.println();
           pw.close();
       } // try
       catch (Exception e) 
       {
           System.out.println("Exception" + e.toString());
       }
       
   } // public void saveControl()
   
   
   /**
    * Prints the weights and labels them by their indices. Index n refers to the
    * layer of the weight. Index j refers to the index of the node the weight edge
    * branches off from. Index k refers to the index of the node the weight edge
    * connects to.
    */
   public void printWeights() 
   {
 
      for (int n = 0; n < numLayers - 1; n++) 
      {
         
         for (int j = 0; j < numNodes[n]; j++) 
         {
            
            for (int k = 0; k < numNodes[n+1]; k++) 
            {
               System.out.println("w" + n + j + k + " " + weights[n][j][k]);
            } // for (int k = 0; k < numNodes[n+1]; k++)
            
         } // for (int j = 0; j < numNodes[n]; j++)
         
      } // for (int n = 0; n < numLayers - 1; n++)

   } // public void printWeights() 
   
   
   /**
    * Saves last known weights to the output file.
    * 
    * @param iteration the number of iterations that need to go by before the weights are saved
    */
   public void saveWeights(int iteration) 
   {
       try 
       {
           PrintWriter pw = new PrintWriter(new FileOutputStream(new File(outputFile), true));
           pw.println("***WEIGHTS***");
           pw.println("iteration = " + iteration);
           
           for (int n = 0; n < numLayers - 1; n++) 
           {
               for (int j = 0; j < numNodes[n]; j++) 
               {
                   for (int i = 0; i < numNodes[n + 1]; i++) 
                   {
                       pw.println( weights[n][j][i]);
                   } // for (int i = 0; i < numNodes[n + 1]; i++)
                   
               } // for (int j = 0; j < numNodes[n]; j++)
               
           } // for (int n = 0; n < numLayers - 1; n++)
           
           pw.println();
           pw.close();
       } // try
       catch (Exception e) 
       {
           System.out.println("Exception" + e.toString());
       }
       
   } // public void saveWeights(int iteration)
   
   
   /**
    * Sets the output file name.
    * 
    * @param filename the name of the file
    */
   public void setOutputFileName (String filename)
   {
       outputFile=filename;
   }
   
   
   /**
    * Reads the user input file and loads the parameters into the appropriate
    * variables.
    * 
    * @param file the file with the user input
    */
   public void readData(String file) 
   {
      
      try 
      {
         BufferedReader reader = new BufferedReader(new FileReader(file));
         StringTokenizer tokenizer = new StringTokenizer(reader.readLine());
        
         mode = tokenizer.nextToken();
         
         tokenizer = new StringTokenizer(reader.readLine());         
         preLoad = tokenizer.nextToken();
         
         tokenizer = new StringTokenizer(reader.readLine());         
         numLayers = Integer.parseInt(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         
         deepestLayer = 0;

         numInputs = Integer.parseInt(tokenizer.nextToken());
         numHiddenLayers1 = Integer.parseInt(tokenizer.nextToken());
         numHiddenLayers2 = Integer.parseInt(tokenizer.nextToken());
         numOutputs = Integer.parseInt(tokenizer.nextToken());
         
         int maxHiddenLayers = Math.max ( numHiddenLayers1 ,numHiddenLayers2);
         
         if (numInputs > maxHiddenLayers && numInputs > numOutputs)
         {
            deepestLayer = numInputs;
         }
         else if (numOutputs > maxHiddenLayers && numOutputs > numInputs)
         {
            deepestLayer = numOutputs;
         }
         else
         {
            deepestLayer = maxHiddenLayers;
         }
         
         tokenizer = new StringTokenizer(reader.readLine());
         numTrainingSets = Integer.parseInt(tokenizer.nextToken());

         allocateMemory(); 
         
         numNodes[0] = numInputs;
         numNodes[1] = numHiddenLayers1;
         numNodes[2] = numHiddenLayers2;
         numNodes[3] = numOutputs;
         
         for (int b = 0; b < numTrainingSets; b++) 
         {
            tokenizer = new StringTokenizer(reader.readLine());
            
            for (int c = 0; c < numInputs + numOutputs; c++) 
            {
               trainingSets[b][c] = Double.parseDouble(tokenizer.nextToken());
            } 
         } // for (int b = 0; b < numTrainingSets; b++)

         tokenizer = new StringTokenizer(reader.readLine());
         maxNumberIterations = Integer.parseInt(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         minError = Double.parseDouble(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         lambda = Double.parseDouble(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         minWeight = Double.parseDouble(tokenizer.nextToken());
         maxWeight = Double.parseDouble(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         writeFrequency = Integer.parseInt(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         outputFile = new String(tokenizer.nextToken());
         tokenizer = new StringTokenizer(reader.readLine());
         readFile = new String(tokenizer.nextToken());
                  
         File myObj = null;
         Scanner myReader = null;
         if (preLoad.equals("R") == false)
         {       
             myObj = new File(readFile);
             myReader = new Scanner(myObj);
             try 
             {
                 while (myReader.hasNextLine()) 
                 {
                   if ( myReader.nextLine().equals("***WEIGHTS***"))
                   {
                       myReader.nextLine(); // To skip the iteration.
                       break;
                   }
                 }
               } // try
             catch (Exception e) 
             {
                 System.out.println("An error occurred.");
                 e.printStackTrace();
             }
         } // if (preLoad.equals("R") == false)
         
         for (int n = 0; n < numLayers - 1; n++) 
         {
            
            for (int j = 0; j < numNodes[n]; j++) 
            {
               
               for (int k = 0; k < numNodes[n+1]; k++) 
               {
                  
                  String nextLine = null;
                  if (preLoad.equals("R") == false)
                  {
                      if (myReader.hasNextLine())
                          nextLine = myReader.nextLine();
                  }
                  
                  if (nextLine == null) 
                  {
                      weights[n][j][k] = (maxWeight - minWeight) * Math.random() + minWeight;
                  } 
                  
                  else if (preLoad.equals("R"))
                  {                      
                      weights[n][j][k] = (maxWeight - minWeight) * Math.random() + minWeight;
                  }
                  else
                  {
                     tokenizer = new StringTokenizer(nextLine);
                     
                     if (!tokenizer.hasMoreTokens()) 
                     {
                         weights[n][j][k] = (maxWeight - minWeight) * Math.random() + minWeight;
                     } 
                     
                     else 
                     {
                        weights[n][j][k] = Double.parseDouble(tokenizer.nextToken());
                     } 
                  } // else
                  
               } // for (int k = 0; k < numNodes[n + 1]; k++)
               
            } // for (int j = 0; j < numNodes[n]; j++)
            
         } // for (int n = 0; n < numLayers - 1; n++)

         if ( myReader != null)
             myReader.close();
      } // try
      catch (Exception e) 
      {
         e.printStackTrace();
      }
      
   } // public void readData(String file) 

   
   /**
    * Runs the network on the training cases.
    */
   public void runTrainingSet() 
   {
      
      System.out.println("RUNNING " + numTrainingSets + " TRAINING SETS");
      System.out.println();

      double totalError = 0.0;
      double maxError = 0.0;

      for (int a = 0; a < numTrainingSets; a++) 
      {
         
         for (int b = 0; b < numInputs; b++) 
         {
            nodeCalculations[0][b] = trainingSets[a][b];
            originalNodeCalculations[0][b] = nodeCalculations[0][b];
         } 

         int caseNum = a + 1;
         System.out.println("CASE " + caseNum);
         
         double currError = 0.0;
         
         calculateNode();
         
         for (int e = 1; e <= numOutputs; e++) 
         {
            
             double expected = trainingSets[a][numInputs + e - 1];

             double calculated = nodeCalculations[numLayers - 1][e-1];
             double error = calculateError(expected, calculated);

             totalError += error;
             currError+=error;
             System.out.println("output: " + calculated  + " expected: "+ expected);
             
         } // for (int e = 1; e <= numOutputs; e++)
         
         System.out.println("error: " + currError);
         
         if ( currError> maxError )
         {
             maxError = currError;         
         }
         
         System.out.println();
         
      } // for (int a = 0; a < numTrainingSets; a++)

      System.out.println("REPORT: ");
      System.out.println();
      System.out.println("total error: " + totalError);
      System.out.println("max error from training sets: " + maxError);
      
   } // public void runTrainingSet()

   
   /**
    * Calculates the value of a node.
    * 
    * @return the calculated value of a node
    */
   public void calculateNode() 
   {
      
      double output = 0.0;
      
      for (int n = 1; n < numLayers; n++) 
      {
         
         for (int j = 0; j < numNodes[n]; j++) 
         {
            output = 0.0;
             
            for (int k = 0; k < numNodes[n - 1]; k++) 
            {
               output += weights[n - 1][k][j] * nodeCalculations[n - 1][k];
            } 
          
            originalNodeCalculations[n][j] =  nodeCalculations[n][j];
            nodeCalculations[n][j] = activationFunction(output);
            
         } // for (int j = 0; j < numNodes[n]; j++)
         
      } // for (int n = 1; n < numLayers; n++)
      
   } // public void calculateNode()
   
   
   /**
    * Runs a specified value through the activation function.
    * 
    * @param x the value to run through the activation function
    * @return the value after it has been run through the activation function
    */
   public double activationFunction(double x) 
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } 
   
   
   /**
    * Runs a specified value through the derivative of the activation function.
    * 
    * @param x the value to run through the derivative of the activation function
    * @return the value after it has been run through the derivative of the
    *         activation function
    */
   public double activationDerivative(double x)
   {
         double v = activationFunction(x);
         return v * (1.0 - v);
   }
   
   
   /**
    * Calculates the error of a calculated output based on the expected value.
    * 
    * @param expected   the expected output
    * @param calculated the output calculated by the network
    * @return the error of a calculated output
    */
   public double calculateError(double expected, double calculated) 
   {
      double diff = expected - calculated;
      return 0.5 * diff * diff;
   }
   
   
   /**
    * Sums the error of all the test cases.
    * 
    * @return the total error of all the test cases
    */
   public double getTotalError()
   {
      double totalError = 0.0;
       
       for (int a = 0; a < numTrainingSets; a++) 
       {     
          
          for (int b = 0; b < numInputs; b++) 
          {
             nodeCalculations[0][b] = trainingSets[a][b];
             originalNodeCalculations[0][b] = nodeCalculations[0][b];
          } 

          calculateNode();
          
          for (int e = 1; e <= numOutputs; e++) 
          {
             
              double expected = trainingSets[a][numInputs + e - 1];
           
              double calculated = nodeCalculations[numLayers - 1][e-1];
              double error = calculateError(expected, calculated);
              totalError += error;         
          } // for (int e = 1; e <= numOutputs; e++)
          
       } // for (int a = 0; a < numTrainingSets; a++)

      return totalError;
      
   } // public double getTotalError()
   
   
   /**
    * Implements the backpropagation algorithm to adjust the weights.
    * 
    * @param trainingSet the index of the training set 
    */
   public void backpropagation(int trainingSet)
   {
       for ( int j = 0; j < numNodes[2] ; j++ )
       {
           omegaJ = 0.0;
           
           for (int i = 0; i < numNodes[3]; i++)
           {
               omegaJ += psiI[i] * weights[2][j][i];
               weights[2][j][i] += lambda * hiddens2[j] * psiI[i]; 
           }
           psiJ[j] = omegaJ * activationDerivative(thetaj[j]); 
       } // for ( int j = 0; j < numNodes[2] ; j++ )
       
       for ( int k = 0 ; k < numNodes[1]; k++)
       {
           omegaK = 0.0;
           for ( int j = 0; j < numNodes[2] ; j ++)
           {
               omegaK += psiJ[j] * weights[1][k][j];
               weights[1][k][j] += lambda *  hiddens1[k] * psiJ[j];
           }
           psiK = omegaK * activationDerivative(thetak[k]); 
           for ( int m = 0 ; m < numNodes[0]; m++)
           {
               weights[0][m][k] += lambda * trainingSets[trainingSet][m] * psiK;               
           }
       } // for ( int k = 0 ; k < numNodes[1]; k++)
       
   } // public void backpropagation(int trainingSet)
   
   
   /**
    * Implements the forward pass of the backpropagation algorithm.
    * 
    * @param trainingSet the index of the training set
    */
   public void forwardPass(int trainingSet)
   {
       double val = 0.0;
       for (int k = 0; k < numNodes[1]; k++)
       {
          val = 0.0;    
          
          for (int m = 0; m < numNodes[0]; m++)
          {
             val += trainingSets[trainingSet][m] * weights[0][m][k];
          } //for (int m = 0; m < n_inputs; m++)
          
          hiddens1[k] = activationFunction(val);
          thetak[k] = val; 
       } //for (int k = 0; k < n_hiddens1; k++)
       
       for (int j = 0; j < numNodes[2]; j++)
       {
          val = 0.0;
                
          for (int k = 0; k < numNodes[1]; k++)
          {
             val += hiddens1[k] * weights[1][k][j];
          } //for (int k = 0; k < n_hiddens1; k++)
          
          thetaj[j] = val;
          hiddens2[j] = activationFunction(val);
       } //for (int j = 0; j < n_hiddens2; j++)
       
       for (int i = 0; i < numNodes[3]; i++)
       {
          val = 0.0;
          
          for (int j = 0; j < numNodes[2]; j++)
          {
             val += hiddens2[j] * weights[2][j][i];
          } 
          
          outputs[i] = activationFunction(val);
          thetai[i] = val;
          omega[i] = trainingSets[trainingSet][numInputs+i] - outputs[i];
          psiI[i] = omega[i] * activationDerivative(thetai[i]);  
       } //for (int i = 0; i < n_outputs; i++)
       
   } // public void forwardPass(int trainingSet)
   
   
   /**
    * Trains the network by adjusting the weights in order to minimize error.
    * Utilizes the backpropagation algorithm to adjust the weights.
    */
   public void train()   
   {
      int numIterations = 0;
      double totalError = getTotalError();
      boolean minErrorReached = false;
      
      long start = System.currentTimeMillis();
      
      while (totalError > minError && numIterations < maxNumberIterations)
      {
         
         int trainingIndex = numIterations % numTrainingSets; 
         forwardPass(trainingIndex);
         backpropagation(trainingIndex);
         
         totalError = getTotalError();
         numIterations++;
         
         if (numIterations % writeFrequency == 0)
         {
             saveControl();
             saveWeights(numIterations);
         }
         
      } // while (maxError > ERROR_THRESHOLD && numIterations++ < MAXIMUM_NUMBER_OF_ITERATION)

      long end = System.currentTimeMillis();
      
      if (totalError < minError) 
      {
         minErrorReached = true;
      }

      System.out.println("REPORT: ");
      System.out.println();
      System.out.print("reason for termination: ");

      if (minErrorReached) 
      {
         System.out.println("error threshold reached");
      } 
      else 
      {
         System.out.println("maximum number of iterations reached");
      }

      System.out.println("error threshold was: " + minError);
      System.out.println("total error was: " + totalError);
      System.out.println("max number of iterations: " + maxNumberIterations);
      System.out.println("number of iterations: " + numIterations);
      System.out.println();
      
      System.out.println ("training time: " + (end - start) + " milliseconds");
      
      saveControl();
      saveWeights(numIterations);
      System.out.println();
      
      runTrainingSet();
      System.out.println("number of iterations: " + numIterations);

   } // public void train()
   
} // public class Perceptron3
