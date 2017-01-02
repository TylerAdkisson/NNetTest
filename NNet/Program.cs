using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class Program
    {
        static double CosineDistance(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
                return -1.0;

            // dist = (a DOT b) / ||a|| ||b||

            double topDot = 0.0;
            for (int i = 0; i < vector1.Length; i++)
            {
                topDot += (vector1[i] * vector2[i]);
            }

            double wallsA = 0.0;
            for (int i = 0; i < vector1.Length; i++)
            {
                wallsA += (vector1[i] * vector1[i]);
            }
            wallsA = Math.Sqrt(wallsA);

            double wallsB = 0.0;
            for (int i = 0; i < vector2.Length; i++)
            {
                wallsB += (vector2[i] * vector2[i]);
            }
            wallsB = Math.Sqrt(wallsB);

            double result = topDot / (wallsA * wallsB);

            return result;
        }
        static void AutoEncodeTest2()
        {
            Layer inputLayer = new Layer(8 * 8);
            Layer hiddenLayer1 = new Layer(24);
            //Layer hiddenLayer2 = new Layer(8 * 8);
            Layer outputLayer = new Layer(8 * 8);

            // Connect layers with random weights
            inputLayer.ConnectForwardFully(hiddenLayer1, true);
            hiddenLayer1.ConnectForwardFully(outputLayer, true);

            //inputLayer.ConnectForwardFully(hiddenLayer1, true);
            //hiddenLayer1.ConnectForwardFully(hiddenLayer2, true);
            //hiddenLayer2.ConnectForwardFully(outputLayer, true);

            // 0.25 / hidden neuron count
            hiddenLayer1.LearningRate = 0.25 / hiddenLayer1.Neurons.Length;//0.028;
            //hiddenLayer2.LearningRate = hiddenLayer1.LearningRate;
            // Output learns 16x faster than hidden
            outputLayer.LearningRate = 16 * hiddenLayer1.LearningRate;//8 * 0.005;//1.0 / hiddenLayer1.Neurons.Length;

            hiddenLayer1.LearningMomentum = 0.0;
            //hiddenLayer2.LearningMomentum = 0.0;
            outputLayer.LearningMomentum = 0.0;

            int imageWidth = 8,
                imageHeight = 8;

            int tileCountX = 376 / imageWidth;
            int tileCountY = 736 / imageHeight;

            int totalCount = tileCountX * tileCountY;

            byte[] pixelSet = new byte[imageWidth * imageHeight];
            double[] pixelNorm = new double[pixelSet.Length];

            double mse = 0.0;

            byte[] totalOutput = new byte[(imageWidth * tileCountX) * (imageHeight * tileCountY)];

            int nextStageIter = 5;

            Random skipRandom = new Random(1);
            int maxIter = int.MaxValue;// 50000;
            bool isTrained = false;
            for (int iter = 0; iter < maxIter && !isTrained; iter++)
            {
                mse = 0.0;
                //Array.Clear(totalOutput, 0, totalOutput.Length);

                hiddenLayer1.LearningRate *= 0.90;
                //if (iter >= nextStageIter)
                //    hiddenLayer2.LearningRate *= 0.90;
                outputLayer.LearningRate *= 0.90;

                if (hiddenLayer1.LearningRate < 0.005)
                    hiddenLayer1.LearningRate = 0.005;

                //if (hiddenLayer2.LearningRate < 0.005)
                //    hiddenLayer2.LearningRate = 0.005;

                if (outputLayer.LearningRate < 0.005)
                    outputLayer.LearningRate = 0.005;

                Console.WriteLine("Epoch: {0:N0}", iter + 1);

                HashSet<int> usedTiles = new HashSet<int>();

                // Train over input set
                for (int ii = 0; ii < totalCount; ii++)
                {
                    //if (skipRandom.NextDouble() < 0.30)
                    //    continue;

                    int i = 0;
                    do
                    {
                        i = skipRandom.Next(0, totalCount);
                    }
                    while (usedTiles.Contains(i));
                    usedTiles.Add(i);

                    bool display = false;
                    //if (i % 1 == 0)
                    //    display = true;

                    //if (i % 1 == 0)
                    //{
                    int tileX = (i % tileCountX) + 1;
                    int tileY = (i / tileCountX) + 1;
                    //tileY = 1;

                    //tileX = 4;
                    //tileY = 5;

                    // Read data
                    pixelSet = File.ReadAllBytes(string.Concat(@"G:\UserData\Tyler\Desktop\CharTest\tiles\TestImage_", tileY.ToString(), "x", tileX.ToString(), ".raw"));
                    //pixelSet = File.ReadAllBytes(string.Concat(@"G:\UserData\Tyler\Desktop\CharTest\eogFocus_20x20.raw"));

                    //tileX = (i % tileCountX) + 1;
                    //tileY = (i / tileCountY) + 1;

                    //pixelSet = File.ReadAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\eogFocus.raw");

                    // Normalize pixel data
                    for (int k = 0; k < pixelSet.Length; k++)
                    {
                        pixelNorm[k] = (pixelSet[k] - 128) / 128.0;
                        //pixelNorm[k] = pixelSet[k] / 255.0;
                    }
                    //}


                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(pixelNorm);

                    // Activate layers
                    hiddenLayer1.Activate();
                    //if (iter >= nextStageIter)
                    //    hiddenLayer2.Activate();
                    outputLayer.Activate();


                    //if (display)
                    //{
                    //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar.raw", pixelSet);


                    byte[] outputPixels = new byte[pixelSet.Length];
                    for (int k = 0; k < outputLayer.Neurons.Length; k++)
                    {
                        double v = outputLayer.Neurons[k].Value;
                        //v *= 255;
                        v *= 128;
                        v += 128;
                        if (v < 0)
                            v = 0;
                        if (v > 255)
                            v = 255;

                        //outputPixels[k] = (byte)((v * 128) + 127);

                        outputPixels[k] = (byte)v;
                    }

                    //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar_output.raw", outputPixels);



                    //for (int y = 0; y < imageHeight; y++)
                    //{
                    //    for (int x = 0; x < imageWidth; x++)
                    //    {
                    //        int srcIndex = (y * imageWidth) + x;

                    //        int xOffset = ((tileX - 1) * imageWidth) + x;
                    //        int index = xOffset + ((y + ((tileY - 1) * imageHeight)) * (imageWidth * tileCountX));

                    //        totalOutput[index] = outputPixels[srcIndex];
                    //    }
                    //}

                    //int temp = tileY;
                    //tileY = tileX;
                    //tileX = temp;

                    //// Copy to output image
                    //for (int y = 0; y < imageHeight; y++)
                    //{
                    //    int adjX = (tileX - 1) * imageWidth;
                    //    int adjY = ((((tileY - 1) * imageHeight) + y) * (tileCountX * imageWidth));

                    //    int scanLineX = (tileX - 1) * imageWidth;
                    //    int imageWidthPx = tileCountX * imageWidth;
                    //    int tileStartY = imageWidthPx * (imageHeight * (tileY - 1));
                    //    int lineY = tileStartY + (y * imageWidthPx);

                    //    Array.Copy(outputPixels, y * imageWidth, totalOutput, adjY + adjX, imageWidth);
                    //}

                    //temp = tileY;
                    //tileY = tileX;
                    //tileX = temp;

                    //}

                    //if (display)
                    //    Console.WriteLine("Target: {0}, Results: {1:N0}", label, i);

                    if (display)
                        Console.WriteLine("Tile: {0}x{1}", tileX, tileY);

                    double localErr = 0.0;
                    for (int o = 0; o < outputLayer.Neurons.Length; o++)
                    {
                        double err = (inputLayer.Neurons[o].Value - outputLayer.Neurons[o].Value);
                        localErr += err * err;
                        //mse += err * err;
                    }
                    mse += localErr / outputLayer.Neurons.Length;
                    if (display)
                    {
                        Console.WriteLine();
                        //Console.WriteLine("MSE: {0,8:F5}", mse / ((iter * imageCount) + (i + 1)));
                        Console.WriteLine("MSE: {0,8:F5}", mse / (i + 1));
                    }


                    // Network training

                    // Network error computation
                    outputLayer.ComputeError(inputLayer.Neurons);
                    //if (iter >= nextStageIter)
                    //    hiddenLayer2.ComputeError();
                    //if (iter < nextStageIter)
                        hiddenLayer1.ComputeError();

                    // Update weights
                    //if (iter < nextStageIter)
                        hiddenLayer1.UpdateWeights();
                    //if (iter >= nextStageIter)
                    //    hiddenLayer2.UpdateWeights();
                    outputLayer.UpdateWeights();
                }

                //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\tiles_output.raw", totalOutput);

                if (iter % nextStageIter != 0)
                    continue;

                //if (iter < nextStageIter * 2 && iter != nextStageIter - 1)
                //    continue;

                //continue;

                Array.Clear(totalOutput, 0, totalOutput.Length);
                int correctCount = 0;
                for (int i = 0; i < totalCount; i++)
                {
                    int tileX = (i % tileCountX) + 1;
                    int tileY = (i / tileCountX) + 1;

                    // Read data
                    pixelSet = File.ReadAllBytes(string.Concat(@"G:\UserData\Tyler\Desktop\CharTest\tiles\TestImage_", tileY.ToString(), "x", tileX.ToString(), ".raw"));

                    //pixelSet = File.ReadAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\eogFocus.raw");

                    // Normalize pixel data
                    for (int k = 0; k < pixelSet.Length; k++)
                    {
                        pixelNorm[k] = (pixelSet[k] - 128) / 128.0;
                        //pixelNorm[k] = pixelSet[k] / 255.0;
                        //pixelNorm[k] = -0.5;
                    }


                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(pixelNorm);

                    // Activate layers
                    hiddenLayer1.Activate();
                    //if (iter >= nextStageIter)
                    //    hiddenLayer2.Activate();
                    outputLayer.Activate();

                    bool display = true;

                    if (display)
                    {
                        //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar.raw", pixelSet);

                        byte[] outputPixels = new byte[pixelSet.Length];
                        for (int k = 0; k < outputLayer.Neurons.Length; k++)
                        {
                            double v = outputLayer.Neurons[k].Value;
                            v *= 128;
                            v += 128;
                            if (v < 0)
                                v = 0;
                            if (v > 255)
                                v = 255;

                            outputPixels[k] = (byte)v;
                        }

                        //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar_output.raw", outputPixels);
                        //File.WriteAllBytes(string.Format(@"G:\UserData\Tyler\Desktop\CharTest\tiles_out\output_{0}x{1}.raw", tileX, tileY), outputPixels);


                        //int temp = tileY;
                        //tileY = tileX;
                        //tileX = temp;

                        //// Copy to output image
                        //for (int y = 0; y < imageHeight; y++)
                        //{
                        //    int adjX = (tileX - 1) * imageWidth;
                        //    int adjY = ((((tileY - 1) * imageHeight) + y) * (tileCountX * imageWidth));

                        //    int scanLineX = (tileX - 1) * imageWidth;
                        //    int imageWidthPx = tileCountX * imageWidth;
                        //    int tileStartY = imageWidthPx * (imageHeight * (tileY - 1));
                        //    int lineY = tileStartY + (y * imageWidthPx);

                        //    Array.Copy(outputPixels, y * imageWidth, totalOutput, adjY + adjX, imageWidth);
                        //}

                        //temp = tileY;
                        //tileY = tileX;
                        //tileX = temp;

                        // Copy to output image
                        for (int y = 0; y < imageHeight; y++)
                        {
                            int adjX = (tileX - 1) * imageWidth;
                            int adjY = ((((tileY - 1) * imageHeight) + y) * (tileCountX * imageWidth));

                            int scanLineX = (tileX - 1) * imageWidth;
                            int imageWidthPx = tileCountX * imageWidth;
                            int tileStartY = imageWidthPx * (imageHeight * (tileY - 1));
                            int lineY = tileStartY + (y * imageWidthPx);

                            Array.Copy(outputPixels, y * imageWidth, totalOutput, adjY + adjX, imageWidth);
                        }
                    }

                    File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\tiles_output.raw", totalOutput);

                    //Console.WriteLine("Tile: {0}x{1}", tileX, tileY);


                    //mse = 0.0;
                    for (int o = 0; o < outputLayer.Neurons.Length; o++)
                    {
                        double err = (inputLayer.Neurons[o].Value - outputLayer.Neurons[o].Value);
                        mse += err * err;
                    }


                    if (display)
                    {
                        //Console.WriteLine();
                        //Console.WriteLine("MSE: {0,8:F5}", mse / ((iter * totalCount) + (i + 1)));
                    }

                }

                //if (iter == nextStageIter - 1)
                //{
                //    Layer oldOutput = outputLayer;
                //    Layer oldHidden2 = hiddenLayer2;

                //    // Our old output layer becomes our second hidden layer
                //    outputLayer = oldHidden2;
                //    hiddenLayer2 = oldOutput;

                //    // Connect second hidden layer
                //    //hiddenLayer1.ConnectForwardFully(hiddenLayer2, true);
                //    hiddenLayer2.ConnectForwardFully(outputLayer, true);
                //}
            }
        }

        static void AutoEncodeTest()
        {
            Layer inputLayer = new Layer(28 * 28);
            Layer hiddenLayer1 = new Layer(32);
            Layer outputLayer = new Layer(28 * 28);

            // Connect layers with random weights
            inputLayer.ConnectForwardFully(hiddenLayer1);
            hiddenLayer1.ConnectForwardFully(outputLayer);

            //inputLayer.ConnectForwardFully(outputLayer);

            //for (int i = 0; i < inputLayer.Neurons.Length; i++)
            //{
            //    Synapse s = new Synapse(inputLayer.Neurons[i], outputLayer.Neurons[i], Neuron.GetRandomWeight());
            //    inputLayer.Neurons[i].Outputs = new[] { s };
            //    outputLayer.Neurons[i].Inputs = new[] { s };
            //}

            hiddenLayer1.LearningRate = 0.01;
            outputLayer.LearningRate = 0.01;

            hiddenLayer1.LearningMomentum = 0.0;
            outputLayer.LearningMomentum = 0.0;


            Stream inputTraining = File.OpenRead(@"G:\UserData\Tyler\Desktop\CharTest\train-images.idx3-ubyte");
            Stream inputTrainingTarget = File.OpenRead(@"G:\UserData\Tyler\Desktop\CharTest\train-labels.idx1-ubyte");

            int imageWidth = 0;
            int imageHeight = 0;
            int imageCount = 0;

            byte[] headerBuffer = new byte[16];
            if (inputTraining.Read(headerBuffer, 0, 16) < 16 ||
                BitConverter.ToInt32(headerBuffer, 0) != 0x03080000)
            {
                Console.WriteLine("Error: Couldn't read image file header.");
                Console.ReadLine();
            }

            imageCount = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 4));
            imageHeight = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 8));
            imageWidth = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 12));

            int labelCount = 0;

            if (inputTrainingTarget.Read(headerBuffer, 0, 8) < 8 ||
                BitConverter.ToInt32(headerBuffer, 0) != 0x01080000)
            {
                Console.WriteLine("Error: Couldn't read label file header.");
                Console.ReadLine();
            }

            labelCount = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 4));

            byte[] pixelSet = new byte[imageWidth * imageHeight];
            double[] pixelNorm = new double[pixelSet.Length];

            long imageFileStart = inputTraining.Position;
            long labelFileStart = inputTrainingTarget.Position;


            double mse = 0.0;

            int maxIter = int.MaxValue;// 50000;
            bool isTrained = false;
            for (int iter = 0; iter < maxIter && !isTrained; iter++)
            {
                inputTraining.Position = imageFileStart;
                inputTrainingTarget.Position = labelFileStart;

                mse = 0.0;

                int label = 0;
                // Train over input set
                for (int i = 0; i < imageCount - 10000; i++)
                {

                    bool display = false;
                    if (i % 1 == 0)
                        display = true;


                    if (i % 1 == 0)
                    {
                        // Read data
                        if (inputTraining.Read(pixelSet, 0, pixelSet.Length) < pixelSet.Length)
                        {
                            Console.WriteLine("Error: Unable to read image data.");
                            Console.ReadLine();
                        }

                        //pixelSet = File.ReadAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\eogFocus.raw");

                        // Normalize pixel data
                        for (int k = 0; k < pixelSet.Length; k++)
                        {
                            //pixelNorm[k] = ((255 - pixelSet[k]) - 128.0) / 128.0;
                            //pixelNorm[k] = (pixelSet[k] - 128.0) / 128.0;
                            //pixelNorm[k] *= 0.8;
                            //pixelNorm[k] = (255 - pixelSet[k]) / 255.0;
                            //pixelNorm[k] = (255 - pixelSet[k]) / 255.0;
                            pixelNorm[k] = pixelSet[k] / 255.0;
                        }

                        // Read target label
                        label = inputTrainingTarget.ReadByte();
                        if (label < 0)
                        {
                            Console.WriteLine("Error: Unable to read label data.");
                            Console.ReadLine();
                        }
                    }


                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(pixelNorm);

                    // Activate layers
                    hiddenLayer1.Activate();
                    //hiddenLayer2.Activate();
                    outputLayer.Activate();


                    if (display)
                    {
                        File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar.raw", pixelSet);


                        byte[] outputPixels = new byte[pixelSet.Length];
                        for (int k = 0; k < outputLayer.Neurons.Length; k++)
                        {
                            double v = outputLayer.Neurons[k].Value;
                            v *= 255;
                            if (v < 0)
                                v = 0;
                            if (v > 255)
                                v = 255;
                            //if (((v * 128) + 127) > 255)
                            //{
                            //}

                            //if (((v * 128) + 127) < -1)
                            //{
                            //}
                            //outputPixels[k] = (byte)((v * 128) + 127);

                            outputPixels[k] = (byte)v;
                        }

                        File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar_output.raw", outputPixels);

                    }

                    if (display)
                        Console.WriteLine("Target: {0}, Results: {1:N0}", label, i);

                    double localErr = 0.0;
                    for (int o = 0; o < outputLayer.Neurons.Length; o++)
                    {
                        if (false && display)
                        {
                            if (outputLayer.Neurons[o].Value > 0.5)
                                Console.ForegroundColor = ConsoleColor.Green;
                            Console.Write(" [{0}: {1,5:F2}]", o, outputLayer.Neurons[o].Value);
                            Console.ResetColor();
                        }

                        double err = (inputLayer.Neurons[o].Value - outputLayer.Neurons[o].Value);
                        localErr += err * err;
                        //mse += err * err;
                    }
                    mse += localErr / outputLayer.Neurons.Length;
                    if (display)
                    {
                        Console.WriteLine();
                        //Console.WriteLine("MSE: {0,8:F5}", mse / ((iter * imageCount) + (i + 1)));
                        Console.WriteLine("MSE: {0,8:F5}", mse / (i + 1));
                    }


                    // Network training

                    // Network error computation
                    outputLayer.ComputeError(inputLayer.Neurons);
                    hiddenLayer1.ComputeError();

                    // Update weights
                    hiddenLayer1.UpdateWeights();
                    outputLayer.UpdateWeights();
                }

                int correctCount = 0;
                for (int i = imageCount - 10000; i < imageCount; i++)
                {
                    // Read data
                    if (inputTraining.Read(pixelSet, 0, pixelSet.Length) < pixelSet.Length)
                    {
                        Console.WriteLine("Error: Unable to read image data.");
                        Console.ReadLine();
                    }


                    // Normalize pixel data
                    for (int k = 0; k < pixelSet.Length; k++)
                    {
                        pixelNorm[k] = pixelSet[k] / 255.0;
                    }

                    // Read target label
                    label = inputTrainingTarget.ReadByte();
                    if (label < 0)
                    {
                        Console.WriteLine("Error: Unable to read label data.");
                        Console.ReadLine();
                    }

                    

                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(pixelNorm);

                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();

                    bool display = true;

                    if (display)
                    {
                        //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar.raw", pixelSet);

                        byte[] outputPixels = new byte[pixelSet.Length];
                        for (int k = 0; k < outputLayer.Neurons.Length; k++)
                        {
                            double v = outputLayer.Neurons[k].Value;
                            v *= 255;
                            if (v < 0)
                                v = 0;
                            if (v > 255)
                                v = 255;

                            outputPixels[k] = (byte)v;
                        }

                        //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar_output.raw", outputPixels);

                    }

                    if (display)
                        Console.WriteLine("Target: {0}, Results: {1:N0}", label, i);

                    double max = 0.0;

                    mse = 0.0;
                    for (int o = 0; o < outputLayer.Neurons.Length; o++)
                    {
                        double err = (inputLayer.Neurons[o].Value - outputLayer.Neurons[o].Value);
                        mse += err * err;
                    }


                    if (display)
                    {
                        Console.WriteLine();
                        Console.WriteLine("MSE: {0,8:F5}", mse / ((iter * imageCount) + (i + 1)));
                    }

                }
            }
        }

        static void NumberTest()
        {
            Layer inputLayer = new Layer(28 * 28);
            Layer hiddenLayer1 = new Layer(15);
            Layer hiddenLayer2 = new Layer(15);
            Layer outputLayer = new Layer(10);

            // Connect layers with random weights
            inputLayer.ConnectForwardFully(hiddenLayer1);
            hiddenLayer1.ConnectForwardFully(hiddenLayer2);
            hiddenLayer2.ConnectForwardFully(outputLayer);

            hiddenLayer1.LearningRate = 0.0125;
            hiddenLayer2.LearningRate = 0.0125;
            outputLayer.LearningRate = 0.0125;

            hiddenLayer1.LearningMomentum = 0.0;
            hiddenLayer2.LearningMomentum = 0.0;
            outputLayer.LearningMomentum = 0.0;


            Stream inputTraining = File.OpenRead(@"G:\UserData\Tyler\Desktop\CharTest\train-images.idx3-ubyte");
            Stream inputTrainingTarget = File.OpenRead(@"G:\UserData\Tyler\Desktop\CharTest\train-labels.idx1-ubyte");

            int imageWidth = 0;
            int imageHeight = 0;
            int imageCount = 0;

            byte[] headerBuffer = new byte[16];
            if (inputTraining.Read(headerBuffer, 0, 16) < 16 ||
                BitConverter.ToInt32(headerBuffer, 0) != 0x03080000)
            {
                Console.WriteLine("Error: Couldn't read image file header.");
                Console.ReadLine();
            }

            imageCount = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 4));
            imageHeight = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 8));
            imageWidth = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 12));

            int labelCount = 0;

            if (inputTrainingTarget.Read(headerBuffer, 0, 8) < 8 ||
                BitConverter.ToInt32(headerBuffer, 0) != 0x01080000)
            {
                Console.WriteLine("Error: Couldn't read label file header.");
                Console.ReadLine();
            }

            labelCount = System.Net.IPAddress.NetworkToHostOrder(BitConverter.ToInt32(headerBuffer, 4));

            byte[] pixelSet = new byte[imageWidth * imageHeight];
            double[] pixelNorm = new double[pixelSet.Length];
            double[] outputData = new double[outputLayer.Neurons.Length];

            long imageFileStart = inputTraining.Position;
            long labelFileStart = inputTrainingTarget.Position;


            double mse = 0.0;

            int maxIter = int.MaxValue;// 50000;
            bool isTrained = false;
            for (int iter = 0; iter < maxIter && !isTrained; iter++)
            {
                inputTraining.Position = imageFileStart;
                inputTrainingTarget.Position = labelFileStart;

                mse = 0.0;

                // Train over input set
                for (int i = 0; i < imageCount - 10000; i++)
                {
                    // Read data
                    if (inputTraining.Read(pixelSet, 0, pixelSet.Length) < pixelSet.Length)
                    {
                        Console.WriteLine("Error: Unable to read image data.");
                        Console.ReadLine();
                    }

                    //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar.raw", pixelSet);

                    // Normalize pixel data
                    for (int k = 0; k < pixelSet.Length; k++)
                    {
                        //pixelNorm[k] = ((255 - pixelSet[k]) - 128.0) / 128.0;
                        //pixelNorm[k] *= 0.8;
                        //pixelNorm[k] = (255 - pixelSet[k]) / 255.0;
                        //pixelNorm[k] = (255 - pixelSet[k]) / 255.0;
                        pixelNorm[k] = pixelSet[k] / 255.0;
                    }

                    // Read target label
                    int label = inputTrainingTarget.ReadByte();
                    if (label < 0)
                    {
                        Console.WriteLine("Error: Unable to read label data.");
                        Console.ReadLine();
                    }

                    // Set output activation target
                    Array.Clear(outputData, 0, outputData.Length);
                    //for (int p = 0; p < outputData.Length; p++)
                    //    outputData[p] = -1.0;

                    // Encode as binary
                    //if (((label >> 0) & 0x01) == 1)
                    //    outputData[0] = 1.0;
                    //if (((label >> 1) & 0x01) == 1)
                    //    outputData[1] = 1.0;
                    //if (((label >> 2) & 0x01) == 1)
                    //    outputData[2] = 1.0;
                    //if (((label >> 3) & 0x01) == 1)
                    //    outputData[3] = 1.0;

                    outputData[label] = 1.0;

                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(pixelNorm);

                    // Activate layers
                    hiddenLayer1.Activate();
                    hiddenLayer2.Activate();
                    outputLayer.Activate();

                    bool display = false;
                    if (i % 1000 == 999)
                        display = true;

                    if (display)
                        Console.WriteLine("Target: {0}, Results: {1:N0}", label, i);

                    double localErr = 0.0;
                    for (int o = 0; o < outputData.Length; o++)
                    {
                        if (display)
                        {
                            if (outputLayer.Neurons[o].Value > 0.5)
                                Console.ForegroundColor = ConsoleColor.Green;
                            Console.Write(" [{0}: {1,5:F2}]", o, outputLayer.Neurons[o].Value);
                            Console.ResetColor();
                        }

                        double err = (outputData[o] - outputLayer.Neurons[o].Value);
                        localErr += err * err;
                        //mse += err * err;
                    }
                    mse += localErr / outputData.Length;
                    if (display)
                    {
                        Console.WriteLine();
                        //Console.WriteLine("MSE: {0,8:F5}", mse / ((iter * imageCount) + (i + 1)));
                        Console.WriteLine("MSE: {0,8:F5}", mse / (i + 1));
                    }
                    

                    // Network training

                    // Network error computation
                    outputLayer.ComputeError(outputData);
                    hiddenLayer2.ComputeError();
                    hiddenLayer1.ComputeError();

                    // Update weights
                    hiddenLayer1.UpdateWeights();
                    hiddenLayer2.UpdateWeights();
                    outputLayer.UpdateWeights();
                }

                //continue;

                int correctCount = 0;
                for (int i = imageCount - 10000; i < imageCount; i++)
                {
                    // Read data
                    if (inputTraining.Read(pixelSet, 0, pixelSet.Length) < pixelSet.Length)
                    {
                        Console.WriteLine("Error: Unable to read image data.");
                        Console.ReadLine();
                    }

                    //File.WriteAllBytes(@"G:\UserData\Tyler\Desktop\CharTest\curChar.raw", pixelSet);

                    // Normalize pixel data
                    for (int k = 0; k < pixelSet.Length; k++)
                    {
                        //pixelNorm[k] = ((255 - pixelSet[k]) - 128.0) / 128.0;
                        //pixelNorm[k] *= 0.8;
                        //pixelNorm[k] = (255 - pixelSet[k]) / 255.0;
                        //pixelNorm[k] = (255 - pixelSet[k]) / 511.0;
                        pixelNorm[k] = pixelSet[k] / 255.0;
                    }

                    // Read target label
                    int label = inputTrainingTarget.ReadByte();
                    if (label < 0)
                    {
                        Console.WriteLine("Error: Unable to read label data.");
                        Console.ReadLine();
                    }

                    // Set output activation target
                    Array.Clear(outputData, 0, outputData.Length);
                    outputData[label] = 1.0;

                    // Encode as binary
                    //if (((label >> 0) & 0x01) == 1)
                    //    outputData[0] = 1.0;
                    //if (((label >> 1) & 0x01) == 1)
                    //    outputData[1] = 1.0;
                    //if (((label >> 2) & 0x01) == 1)
                    //    outputData[2] = 1.0;
                    //if (((label >> 3) & 0x01) == 1)
                    //    outputData[3] = 1.0;

                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(pixelNorm);

                    // Activate layers
                    hiddenLayer1.Activate();
                    hiddenLayer2.Activate();
                    outputLayer.Activate();

                    bool display = true;

                    if (display)
                        Console.WriteLine("Target: {0}, Results: {1:N0}", label, i);

                    int maxIndex = 0;
                    double max = 0.0;

                    for (int o = 0; o < outputData.Length; o++)
                    {
                        if (outputLayer.Neurons[o].Value > max)
                        {
                            maxIndex = o;
                            max = outputLayer.Neurons[o].Value;
                        }
                    }

                    for (int o = 0; o < outputData.Length; o++)
                    {
                        if (display)
                        {
                            //if (outputLayer.Neurons[o].Value > 0.5)
                            if(o == maxIndex)
                                Console.ForegroundColor = ConsoleColor.Green;
                            Console.Write(" [{0}: {1,5:F2}]", o, outputLayer.Neurons[o].Value);
                            Console.ResetColor();
                        }

                        double err = (outputData[o] - outputLayer.Neurons[o].Value);
                        mse += err * err;
                    }

                    //int maxIndex = 0;
                    //if (outputLayer.Neurons[0].Value > 0.5)
                    //    maxIndex |= 0x01;
                    //if (outputLayer.Neurons[1].Value > 0.5)
                    //    maxIndex |= 0x02;
                    //if (outputLayer.Neurons[2].Value > 0.5)
                    //    maxIndex |= 0x04;
                    //if (outputLayer.Neurons[3].Value > 0.5)
                    //    maxIndex |= 0x08;

                    if (label == maxIndex)
                        correctCount++;

                    if (display)
                    {
                        Console.WriteLine();
                        Console.WriteLine("MSE: {0,8:F5}", mse / ((iter * imageCount) + (i + 1)));
                        Console.WriteLine("Result: {0} = {1} [{2}]", label, maxIndex, label == maxIndex ? "TRUE" : "FALSE");
                        Console.WriteLine("Percent: {0:P2}", correctCount / ((i - (imageCount - 10000.0)) + 1));
                    }

                }

                continue;

                //if (iter < maxIter - 1)
                if (iter % 5000 < 4999)
                    continue;

                // Validate against test set
                Console.WriteLine("Iteration {0,9:N0}, Validation test: ", iter + 1);
                double error = 0.0;
                for (int i = 0; i < 10000; i++)
                {
                    //// Supply input data
                    //inputLayer.SetValues(inputSet[i]);

                    //// Activate layers
                    //hiddenLayer1.Activate();
                    //outputLayer.Activate();

                    //// Input values
                    //for (int k = 0; k < inputSet[i].Length; k++)
                    //{
                    //    double inputValue = inputSet[i][k];

                    //    // For functions that input in rads, we want to display in degrees
                    //    inputValue = (inputValue * Math.PI) + Math.PI;
                    //    inputValue *= (180 / Math.PI);

                    //    Console.Write("{0,8:F4} ", inputValue);
                    //}

                    //Console.Write("= ");

                    //for (int k = 0; k < targetSet[i].Length; k++)
                    //{
                    //    Console.Write("[{0,10:F6} Target: {1,8:F4} Err: {2,10:F6}]",
                    //        outputLayer.Neurons[k].Value,
                    //        targetSet[i][k],
                    //        targetSet[i][k] - outputLayer.Neurons[k].Value);

                    //    double localError = targetSet[i][k] - outputLayer.Neurons[k].Value;
                    //    error += localError * localError;
                    //}
                    //Console.WriteLine();

                }

                Console.WriteLine("MSE: {0:F6}", error);

                if (error <= 0.0008)
                    isTrained = true;

                //if (Console.ReadKey(true).Key == ConsoleKey.Q)
                //{
                //    break;
                //}
            }
        }

        static Random _shuffleRandom = new Random();
        internal static void Shuffle2<T>(T[] array1, T[] array2)
        {
            for (int i = 0; i < array1.Length / 2; i++)
            {
                int sourceElement = _shuffleRandom.Next(0, array1.Length);
                int destinationElement = _shuffleRandom.Next(0, array1.Length);

                // First array
                T temp = array1[destinationElement];
                array1[destinationElement] = array1[sourceElement];
                array1[sourceElement] = temp;

                // Second array
                temp = array2[destinationElement];
                array2[destinationElement] = array2[sourceElement];
                array2[sourceElement] = temp;
            }
        }
        
        static void Test2()
        {
            Layer inputLayer = new Layer(1);
            Layer hiddenLayer1 = new Layer(10);
            Layer outputLayer = new Layer(2);

            // Connect layers with random weights
            inputLayer.ConnectForwardFully(hiddenLayer1);
            hiddenLayer1.ConnectForwardFully(outputLayer);

            hiddenLayer1.LearningRate = 0.1;
            outputLayer.LearningRate = 0.1;

            hiddenLayer1.LearningMomentum = 0.0;
            outputLayer.LearningMomentum = 0.0;

            // Bias node defaults to a value of 1.0
            //// Set bias nodes' values
            //inputLayer.Bias.Value = 1.0;
            //hiddenLayer1.Bias.Value = 1.0;

            // Train
            // Input data set
            int totalDataCount = 48;
            int trainingDataCount = 40;
            double[][] inputSet = new double[totalDataCount][];
            //inputSet[0] = new[] { -1.0, -1.0 };
            //inputSet[1] = new[] { -1.0, +1.0 };
            //inputSet[2] = new[] { +1.0, -1.0 };
            //inputSet[3] = new[] { +1.0, +1.0 };

            // Output data set for input data set
            double[][] targetSet = new double[inputSet.Length][];
            //targetSet[0] = new[] { -1.0 };
            //targetSet[1] = new[] { +1.0 };
            //targetSet[2] = new[] { +1.0 };
            //targetSet[3] = new[] { -1.0 };

            // Fill with data for Cos
            double radsPerStep = (Math.PI * 2) / inputSet.Length;
            for (int i = 0; i < inputSet.Length; i++)
            {
                double rads = (i * radsPerStep);
                double normalized = (rads - Math.PI) / Math.PI;

                inputSet[i] = new[] { normalized };
                targetSet[i] = new[] { Math.Cos(rads), Math.Sin(rads) };
            }

            double[][] trainInputSet = new double[trainingDataCount][];
            //Array.Copy(inputSet, trainInputSet, trainingDataCount);

            double[][] trainTargetSet = new double[trainingDataCount][];
            //Array.Copy(targetSet, trainTargetSet, trainingDataCount);

            // Select random items from full data set for training
            Random dataSetRandom = new Random(1 /*seed*/);
            HashSet<int> selectedIndexes = new HashSet<int>();
            for (int i = 0; i < trainingDataCount; i++)
            {
                int index = 0;
                do
                {
                    index = dataSetRandom.Next(0, totalDataCount);
                } while (selectedIndexes.Contains(index));
                selectedIndexes.Add(index);

                trainInputSet[i] = inputSet[index];
                trainTargetSet[i] = targetSet[index];
            }

            Stream output = File.Create(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "nntest.raw"));

            int maxIter = int.MaxValue;// 50000;
            bool isTrained = false;
            for (int iter = 0; iter < maxIter && !isTrained; iter++)
            {
                // Jumble input data order as to not get into a loop we can't train
                Shuffle2(trainInputSet, trainTargetSet);

                // Train over input set
                for (int i = 0; i < trainInputSet.Length; i++)
                {
                    // Network activation

                    // Supply input data
                    inputLayer.SetValues(trainInputSet[i]);

                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();


                    // Network training

                    // Network error computation
                    outputLayer.ComputeError(trainTargetSet[i]);
                    hiddenLayer1.ComputeError();

                    // Update weights
                    hiddenLayer1.UpdateWeights();
                    outputLayer.UpdateWeights();
                }

                //if (iter < maxIter - 1)
                if(iter % 5000 < 4999)
                    continue;

                // Validate against test set
                Console.WriteLine("Iteration {0,9:N0}, Validation test: ", iter + 1);
                double error = 0.0;
                for (int i = 0; i < inputSet.Length; i++)
                {
                    // Supply input data
                    inputLayer.SetValues(inputSet[i]);

                    // Activate layers
                    hiddenLayer1.Activate();
                    outputLayer.Activate();

                    // Input values
                    for (int k = 0; k < inputSet[i].Length; k++)
                    {
                        double inputValue = inputSet[i][k];

                        // For functions that input in rads, we want to display in degrees
                        inputValue = (inputValue * Math.PI) + Math.PI;
                        inputValue *= (180 / Math.PI);

                        Console.Write("{0,8:F4} ", inputValue);
                    }

                    Console.Write("= ");

                    for (int k = 0; k < targetSet[i].Length; k++)
                    {
                        Console.Write("[{0,10:F6} Target: {1,8:F4} Err: {2,10:F6}]",
                            outputLayer.Neurons[k].Value,
                            targetSet[i][k],
                            targetSet[i][k] - outputLayer.Neurons[k].Value);

                        double localError = targetSet[i][k] - outputLayer.Neurons[k].Value;
                        error += localError * localError;
                    }
                    Console.WriteLine();

                    //output.Write(BitConverter.GetBytes((float)outputLayer.Neurons[0].Value), 0, 4);
                    //output.Write(BitConverter.GetBytes((float)outputLayer.Neurons[1].Value), 0, 4);
                    //output.Write(BitConverter.GetBytes((float)targetSet[i][0]), 0, 4);
                    //output.Write(BitConverter.GetBytes((float)targetSet[i][1]), 0, 4);
                    //output.Write(BitConverter.GetBytes((float)(targetSet[i][0] - outputLayer.Neurons[0].Value)), 0, 4);
                }

                Console.WriteLine("MSE: {0:F6}", error);
                output.Write(BitConverter.GetBytes((float)error), 0, 4);

                if (error <= 0.0008)
                    isTrained = true;

                //if (Console.ReadKey(true).Key == ConsoleKey.Q)
                //{
                //    break;
                //}
            }

            output.Dispose();

            Console.Write("Press enter to exit...");
            Console.ReadLine();
        }

        static Random random = new Random(3);
        //static double lastOut = 1.0;
        static double GetRandomWeight()
        {
            double value = (random.NextDouble() * 2.0) - 1.0;
            //if ((value < 0 && lastOut < 0) ||
            //    (value >= 0 && lastOut >= 0))
            //    value = -value;

            //lastOut = value;

            //Console.WriteLine("Weight: {0:F8}", value);
            return value;
        }

        static void Test1()
        {
            Stopwatch timer = new Stopwatch();
            int seed = 0;
        start:
            timer.Restart();
            Console.Write("Seed {0}: ", seed);
            random = new Random();

            // Layers
            Neuron[] inputs = new Neuron[2];
            Neuron[] hidden1 = new Neuron[9]; // 9 for Sin/Cos, 3 for XOR
            Neuron[] outputs = new Neuron[1];

            // Populate layers
            // Input layer
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = new Neuron();
            }

            // Hidden layer 1
            for (int i = 0; i < hidden1.Length; i++)
            {
                hidden1[i] = new Neuron();

                // Link with all inputs
                hidden1[i].Inputs = new Synapse[inputs.Length];

                for (int k = 0; k < inputs.Length; k++)
                {
                    // Resize input's output link list to allow room for us
                    Synapse[] newLinks = new Synapse[inputs[k].Outputs.Length + 1];
                    Array.Copy(inputs[k].Outputs, 0, newLinks, 0, inputs[k].Outputs.Length);
                    inputs[k].Outputs = newLinks;

                    // Create new link
                    inputs[k].Outputs[newLinks.Length - 1] = hidden1[i].Inputs[k] =
                        new Synapse(inputs[k], hidden1[i], GetRandomWeight());
                }
            }

            // Output layer
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = new Neuron();

                // Link with all hidden layer neurons
                outputs[i].Inputs = new Synapse[hidden1.Length];

                for (int k = 0; k < hidden1.Length; k++)
                {
                    // Resize input's output link list to allow room for us
                    Synapse[] newLinks = new Synapse[hidden1[k].Outputs.Length + 1];
                    Array.Copy(hidden1[k].Outputs, 0, newLinks, 0, hidden1[k].Outputs.Length);
                    hidden1[k].Outputs = newLinks;

                    // Create new link
                    hidden1[k].Outputs[newLinks.Length - 1] = outputs[i].Inputs[k] =
                        new Synapse(hidden1[k], outputs[i], GetRandomWeight());
                }
            }

            // Set bias nodes
            inputs[inputs.Length - 1].Value = 1.0;
            hidden1[hidden1.Length - 1].Value = 1.0;

            // Train
            // Input data set
            double[][] inputSet = new double[32][];
            //inputSet[0] = new[] { 0.0, 0.0 };
            //inputSet[1] = new[] { 0.0, 1.0 };
            //inputSet[2] = new[] { 1.0, 0.0 };
            //inputSet[3] = new[] { 1.0, 1.0 };

            // Fill with data for Sin/Cos
            double degPerStep = 360.0 / inputSet.Length;
            double radPerStep = (2 * Math.PI) / inputSet.Length;
            for (int i = 0; i < inputSet.Length; i++)
            {
                //double val = i * degPerStep;
                //val -= 180;
                //val /= 180;
                double val = i * radPerStep;
                val -= Math.PI;
                val /= Math.PI;
                inputSet[i] = new[] { val/*, 0.0*/ };
            }

            //0.01745329251994329576923690768489
            //inputSet[0] = new[] { Math.Sin(0.0), 0.0 }; // 0 Degrees
            //inputSet[1] = new[] { Math.Sin(1.570796), 0.0 }; // 90 Degrees
            //inputSet[2] = new[] { Math.Sin(3.1415926), 0.0 }; // 180 Degrees
            //inputSet[3] = new[] { Math.Sin(4.7123889), 0.0 }; // 270 Degrees

            // Output data set for input data set
            double[][] targetSet = new double[inputSet.Length][];
            //targetSet[0] = new[] { 0.0 };
            //targetSet[1] = new[] { 1.0 };
            //targetSet[2] = new[] { 1.0 };
            //targetSet[3] = new[] { 0.0 };

            // Fill with data for Sin/Cos
            for (int i = 0; i < targetSet.Length; i++)
            {
                double val = Math.Sin(i * radPerStep);
                targetSet[i] = new[] { val };
            }


            double learningRate = 0.1;
            double momentum = 0.1;

            int inputSetIndex = 0;

            double errSquares = 0.0;
            double mse = 1.0;

            int goodCount = 0;
            int p = 0;

            //Stream output = File.Create(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "nntest.raw"));

            for (p = 0; /*p < 20000*/ mse > 0.000005 || goodCount < 20; p++)
            {
                if (mse > 0.0005)
                    goodCount = 0;
                else
                    goodCount++;

                //inputSetIndex = (inputSetIndex + 1) % inputSet.Length;

                // Presenting training data randomly is very important
                // Presenting data in a random order reduces the likelyhood of the network never converging
                inputSetIndex = random.Next(0, inputSet.Length);

                // Data input phase
                // Set inputs to data set
                for (int i = 0; i < inputs.Length - 1; i++)
                {
                    inputs[i].Value = inputSet[inputSetIndex][i];
                    //Console.Write("{0,6:F2} ", inputs[i].Value);
                }
                //Console.Write(" = ");


                // Activation phase
                // You don't activate input neurons, they just hold input data
                //// Activate inputs
                //for (int i = 0; i < inputs.Length; i++)
                //{
                //    inputs[i].Activate();
                //}

                // Activate hidden layer 1
                for (int i = 0; i < hidden1.Length - 1; i++)
                {
                    hidden1[i].Activate();
                }

                // Activate output layer
                for (int i = 0; i < outputs.Length; i++)
                {
                    outputs[i].Activate();
                    //Console.Write("{0,8:F5} ", outputs[i].Value);
                }


                // Error computation phase
                // Compute delta value of output
                double[] deltasOut = new double[outputs.Length];
                for (int i = 0; i < outputs.Length; i++)
                {
                    double error = targetSet[inputSetIndex][i] - outputs[i].Value;
                    //double error = outputs[i].Value - targetSet[inputSetIndex][i];
                    deltasOut[i] = outputs[i].ErrorDerivFunction(outputs[i].Value) * error;

                    errSquares += (error * error);
                    errSquares *= 0.5;
                    mse = errSquares / (i + 1);

                    //Console.Write("Err: {0,8:F5} ", error);
                    //Console.Write("MSE: {0,8:F5}", mse);
                    //output.Write(BitConverter.GetBytes(mse), 0, 8);
                }
                //Console.WriteLine();

                // Compute delta value of hidden layer 1
                double[] deltasHid1 = new double[hidden1.Length];
                for (int i = 0; i < hidden1.Length; i++)
                {
                    double error = 0.0;

                    // Sum error values pulled from all of the neurons we output to
                    // Each delta value is multiplied by the weight of the synapse that
                    //   links us to that specific output neuron
                    for (int k = 0; k < hidden1[i].Outputs.Length; k++)
                    {
                        error += deltasOut[k] * hidden1[i].Outputs[k].Weight;
                    }

                    // Our delta value is the derivitive of our activation value multiplied by
                    //   the summed error value
                    deltasHid1[i] = hidden1[i].ErrorDerivFunction(hidden1[i].Value) * error;
                }


                // Weight adjustment phase
                // Adjust input weights for all neurons in hidden layer 1
                for (int i = 0; i < hidden1.Length; i++)
                {
                    // Adjust all hidden layer inputs
                    for (int k = 0; k < hidden1[i].Inputs.Length; k++)
                    {
                        // Add a percentage (the learning rate) of the derived error value
                        //   (computed above) multiplied by the input value to the synapse weight
                        Synapse synapse = hidden1[i].Inputs[k];
                        double change = (deltasHid1[i] * synapse.Input.Value);
                        synapse.Weight += (learningRate * change) + (momentum * synapse.LastChange);
                        synapse.LastChange = change;
                    }
                }

                // Adjust input weights for all neurons in output layer
                for (int i = 0; i < outputs.Length; i++)
                {
                    // Adjust all output layer inputs
                    for (int k = 0; k < outputs[i].Inputs.Length; k++)
                    {
                        // Add a percentage (the learning rate) of the derived error value
                        //   (computed above) multiplied by the input value to the synapse weight
                        Synapse synapse = outputs[i].Inputs[k];
                        double change = (deltasOut[i] * synapse.Input.Value);
                        synapse.Weight += (learningRate * change) + (momentum * synapse.LastChange);
                        synapse.LastChange = change;
                    }
                }

                // Training lap complete
            }

            //output.Dispose();

            timer.Stop();
            Console.WriteLine("{0:F3}", timer.Elapsed.TotalSeconds);
            seed++;

            //goto start;

            // Test network
            Console.WriteLine("Network Test: {0:N0} loops", p);
            for (inputSetIndex = 0; inputSetIndex < inputSet.Length; inputSetIndex++)
            {
                // Data input phase
                // Set inputs to data set
                for (int i = 0; i < inputs.Length - 1; i++)
                {
                    inputs[i].Value = inputSet[inputSetIndex][i];
                    double deg = ((inputs[i].Value * Math.PI) + Math.PI) * 57.295779513082320876798154814105;
                    double inputValue = deg; // inputs[i].Value;
                    Console.Write("{0,6:F2} ", inputValue);
                }
                Console.Write(" = ");

                // Activate hidden layer 1
                for (int i = 0; i < hidden1.Length - 1; i++)
                {
                    hidden1[i].Activate();
                }

                // Activate output layer
                for (int i = 0; i < outputs.Length; i++)
                {
                    outputs[i].Activate();
                    Console.Write("{0,8:F5}, Target: {1,8:F5}, Err: {2,8:F5}",
                        outputs[i].Value,
                        targetSet[inputSetIndex][i],
                        targetSet[inputSetIndex][i] - outputs[i].Value);

                    //output.Write(BitConverter.GetBytes((float)outputs[i].Value), 0, 4);
                    //output.Write(BitConverter.GetBytes((float)targetSet[inputSetIndex][i]), 0, 4);
                }
                Console.WriteLine();
            }

            //output.Dispose();

            Console.ReadLine();

            goto start;

        }

        static void Main(string[] args)
        {
            //AutoEncodeTest2();
            //AutoEncodeTest();

            //NumberTest();
            //Test2();

            //Test1();
            ConditionalSwitch.MainVoid(args);
        }
    }
}
