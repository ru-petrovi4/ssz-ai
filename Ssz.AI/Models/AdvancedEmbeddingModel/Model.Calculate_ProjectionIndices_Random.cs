using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model
    {
        public void Calculate_ProjectionIndices_Random(ILoggersSet loggersSet)
        { 
            var totalStopwatch = Stopwatch.StartNew();

            var r = new Random();

            var wordsProjectionIndices = new int[Words_RU.Count];

            foreach (int wordIndex in Enumerable.Range(0, wordsProjectionIndices.Length))
            {
                wordsProjectionIndices[wordIndex] = r.Next(NewVectorLength);
            }
            //int[] pointProjectionArray = Enumerable.Repeat(-1, Cortex.Array.Length).ToArray();
            //for (int i = 0; i < BooleanVectorLength; i += 1)
            //{
            //    for (; ; )
            //    {
            //        ref int e = ref pointProjectionArray[r.Next(pointProjectionArray.Length)];
            //        if (e == -1)
            //        {
            //            e = i;
            //            break;
            //        }                    
            //    }
            //}

            ProjectionOptimization_Algorithm_Random.WordsProjectionIndices = wordsProjectionIndices;


            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("Calculate_ProjectionIndices_Random totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }
    }    
}