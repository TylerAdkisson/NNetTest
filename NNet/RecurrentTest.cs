using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class RecurrentTest
    {
        public static void MainVoid(string[] args)
        {
            //Recurrent1();

            // English text (ASCII/UTF-8):
            // 32 - Space
            // 48..57: 0..9
            // 65..90: Uppercase A..Z
            //
            // Total neurons required for input/output: 37
        }

        /// <summary>
        /// Demonstrates a toy recurrent network, spelling the word 'hello'
        /// </summary>
        static void Recurrent1()
        {
            //Layer inputLayer = new Layer(1);
            Layer hiddenLayer1 = new Layer(5, Neuron.ActivationType.Tanh, Layer.LearningType.Adagrad);
            Layer prevHiddenLayer1 = new Layer(5);
            Layer outputLayer = new Layer(4, Neuron.ActivationType.Tanh, Layer.LearningType.Adagrad);

            // Connect layers with random weights
            //inputLayer.ConnectForwardFully(hiddenLayer1);
            prevHiddenLayer1.ConnectForwardFully(hiddenLayer1);
            hiddenLayer1.ConnectForwardFully(outputLayer);

            hiddenLayer1.LearningRate = 0.3;
            outputLayer.LearningRate = 0.3;

            // Momentum and weight decay do not apply to the adagrad learning method
            hiddenLayer1.LearningMomentum = 0.95;
            outputLayer.LearningMomentum = 0.95;

            hiddenLayer1.WeightDecay = 0.000;
            outputLayer.WeightDecay = 0.000;

            // Change the mode of some of the hidden layer's activation functions
            //hiddenLayer1.Neurons[0].SetNeuronType(Neuron.ActivationType.Tanh);

            int trainingDataCount = 5;
            int totalDataCount = 5;
            //double[][] inputSet = new double[totalDataCount][];
            //double[][] trainInputSet = new double[trainingDataCount][];
            double[][] targetSet = new double[totalDataCount][];
            double[][] trainTargetSet = new double[trainingDataCount][];
            Random r = new Random(11);

            //// Create training data set (simple if-then-like formula)
            //for (int i = 0; i < inputSet.Length; i++)
            //{
            //    inputSet[i] = new[] { 1.0, 0.0, 0.0, 0.0 };// * 2.0 - 1.0 };
            //    targetSet[i] = new[] { targetRodPercent };// * 2.0 - 1.0 };
            //}

            // Target sequence: h, e, l, l, o
            targetSet[0] = new[] { 1.0, 0.0, 0.0, 0.0 };// * 2.0 - 1.0 };
            targetSet[1] = new[] { 0.0, 1.0, 0.0, 0.0 };// * 2.0 - 1.0 };
            targetSet[2] = new[] { 0.0, 0.0, 1.0, 0.0 };// * 2.0 - 1.0 };
            targetSet[3] = new[] { 0.0, 0.0, 1.0, 0.0 };// * 2.0 - 1.0 };
            targetSet[4] = new[] { 0.0, 0.0, 0.0, 1.0 };// * 2.0 - 1.0 };

            //trainInputSet = inputSet;
            trainTargetSet = targetSet;


            bool isTrained = false;
            for (int iter = 0; iter < 2000 && !isTrained; iter++)
            {
                // Reset inital hidden to 0
                prevHiddenLayer1.SetValues(new[] { 0.0, 0.0, 0.0, 0.0, 0.0 });

                // Train over input set
                for (int i = 0; i < trainingDataCount; i++)
                {
                    // Network activation

                    // Supply input data
                    //inputLayer.SetValues(trainInputSet[i]);
                    // We have no input data with this network

                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();

                    // Copy hidden to prev hidden
                    prevHiddenLayer1.SetValues(hiddenLayer1.Neurons);


                    // Network training

                    // Network error computation
                    outputLayer.ComputeError(trainTargetSet[i]);
                    hiddenLayer1.ComputeError();

                    // Update weights
                    hiddenLayer1.UpdateWeights();
                    outputLayer.UpdateWeights();
                }

                // Validate sequence
                // Reset inital hidden to 0
                prevHiddenLayer1.SetValues(new[] { 0.0, 0.0, 0.0, 0.0, 0.0 });

                Console.WriteLine("    {0,7} {1,7} {2,7} {3,7}", "h", "e", "l", "o");
                for (int i = 0; i < trainingDataCount; i++)
                {
                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();

                    // Copy hidden to prev hidden
                    prevHiddenLayer1.SetValues(hiddenLayer1.Neurons);

                    Console.Write("[{0}]", i);
                    int maxV = 0;
                    for (int p = 0; p < 4; p++)
                    {
                        if (outputLayer.Neurons[p].Value > outputLayer.Neurons[maxV].Value)
                            maxV = p;
                    }

                    for (int p = 0; p < 4; p++)
                    {
                        if (p == maxV)
                            Console.ForegroundColor = ConsoleColor.Green;
                        Console.Write(" {0,7:F4}", outputLayer.Neurons[p].Value);
                        Console.ResetColor();
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();

                
            }
        }


        static char NN2ToChar(Neuron[] values)
        {
            int greatestIndex = 0;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i].Value > values[greatestIndex].Value)
                    greatestIndex = i;
            }

            if (greatestIndex == 0)
                return ' ';
            else if (greatestIndex >= 1 && greatestIndex <= 9)
                return (char)(48 + (greatestIndex - 1));
            else if (greatestIndex >= 10 && greatestIndex <= 90)
                return (char)(65 + (greatestIndex - 10));
            return '?';
        }

        static char NN2ToChar(double[] values)
        {
            int greatestIndex = 0;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > values[greatestIndex])
                    greatestIndex = i;
            }

            if (greatestIndex == 0)
                return ' ';
            else if (greatestIndex >= 1 && greatestIndex <= 9)
                return (char)(48 + (greatestIndex - 1));
            else if (greatestIndex >= 10 && greatestIndex <= 90)
                return (char)(65 + (greatestIndex - 10));
            return '?';
        }

        static int CharToNN2(char chr)
        {
            int index = 0;
            if (chr == 32)
                index = 0;
            else if (chr >= 48 && chr <= 57)
                index = 1 + (chr - 48);
            else if (chr >= 65 && chr <= 90)
                index = 10 + (chr - 65);
            else if (chr >= 97 && chr <= 122)
                index = 10 + (chr - 97);
            else
                return -1;

            return index;
        }
    }
}
