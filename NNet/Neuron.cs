using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class Neuron
    {
        private static Random _weightRandom;
        private static readonly double _weightRange = 1.0;//0.000125944584;//0.00125944584382871536523929471033;//2.0;


        public Synapse[] Inputs;
        public Synapse[] Outputs;
        public double Value;
        public double ErrorDelta;


        public Neuron()
        {
            Inputs = new Synapse[0];
            Outputs = new Synapse[0];
            Value = 0;
        }


        public static double Sigmoid(double x)
        {
            //return 1.7159 * Math.Tanh(0.66666667 * x);
            return Math.Tanh(x);
            //return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double DerivSigmoid(double x)
        {
            //return 0.66666667 * (1.7159 - (x * x));

            //1.14393 * (1- tanh^2 ( 2/3 * x))
            //double tanh2 = Math.Tanh(0.66666667 * x);
            //return 1.14393 * (1 - (tanh2 * tanh2));

            return 1.0 - (x * x);
            //return Sigmoid(x) * (1.0 - Sigmoid(x));
        }

        public static double GetRandomWeight()
        {
            if (_weightRandom == null)
                _weightRandom = new Random(1);

            return (_weightRandom.NextDouble() * (_weightRange * 2)) - _weightRange;
        }

        /// <summary>
        /// Take input values and updates our value
        /// </summary>
        public void Activate()
        {
            // Consider values of all of our inputs
            double sum = 0.0;
            for (int i = 0; i < Inputs.Length; i++)
            {
                // Sum all input values, multiplied by the link weights
                //double adj = Sigmoid(Inputs[i].Input.Value * Inputs[i].Weight);
                //sum += adj;
                sum += Inputs[i].Input.Value * Inputs[i].Weight;
            }

            //while (Math.Abs(sum) >= 1.0)
            //    sum /= 10.0;

            //sum /= 10.0;
            //sum /= (Inputs.Length / 100.0);

            // Differentiate sum with sigmoid function (could be tanh)
            Value = Sigmoid(sum);

            //int intValue = (int)(Value * 127);
            //Value = intValue / 127.0;
        }

        public override string ToString()
        {
            return string.Format("Value: {0:F5}, Inputs: {1:N0}, Outputs: {2:N0}",
                Value,
                Inputs.Length,
                Outputs.Length);
        }
    };

}
