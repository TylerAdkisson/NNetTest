using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNet
{
    class Synapse
    {
        public readonly Neuron Input;
        public readonly Neuron Output;
        public double Weight;
        public double LastChange;


        public Synapse(Neuron input, Neuron output, double weight)
        {
            Input = input;
            Output = output;
            Weight = weight;
        }

        public override string ToString()
        {
            return string.Format("Weight: {0:F8}, Change: {1:F8}", Weight, LastChange);
        }
    };
}
