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
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model
    {
        public void Calculate_ProjectionIndices_Variant3(Clusterization_Algorithm clusterization_Algorithm, ILoggersSet loggersSet)
        {
            var totalStopwatch = Stopwatch.StartNew();

            var r = new Random();

            ProjectionOptimization_Algorithm_Variant3.WordsProjectionIndices = new int[Words_RU.Count];
            //LoadFromFile_ProjectionIndices(ProjectionOptimization_Algorithm_Variant3, "ProjectionOptimization.bin", _loggersSet);
            var wordsProjectionIndices = ProjectionOptimization_Algorithm_Variant3.WordsProjectionIndices;
            //Random initial hash
            foreach (int wordIndex in Enumerable.Range(0, wordsProjectionIndices.Length))
            {
                wordsProjectionIndices[wordIndex] = r.Next(NewVectorLength);
            }                                      

            NewVectorsAndMatrices newVectorsAndMatrices = new();
            newVectorsAndMatrices.Initialize(Words_RU.Count);
            newVectorsAndMatrices.InitializeTemp(clusterization_Algorithm, Words_RU, ProxWordsOldMatrix);
            newVectorsAndMatrices.Calculate_Full(Words_RU, wordsProjectionIndices, loggersSet);

            Buffers buffers = new(newVectorsAndMatrices);                                              

            for (int i = 0; i < 1; i += 1)
            {
                var words_RandomOrder = new List<Word>(Words_RU.Count);
                foreach (var word in Words_RU)
                {
                    word.Temp_Flag = false;
                }
                for (int index = 0; index < Words_RU.Count; index += 1)
                {
                    for (; ; )
                    {
                        var word = Words_RU[r.Next(Words_RU.Count)];
                        if (word.Temp_Flag)
                            continue;

                        words_RandomOrder.Add(word);
                        word.Temp_Flag = true;
                        break;
                    }
                }

                var stopwatch = Stopwatch.StartNew();
                float energy;
                int wordN = 0;
                foreach (Word word in words_RandomOrder)
                {
                    energy = OptimizeWordProjection(word, wordsProjectionIndices, newVectorsAndMatrices, buffers, loggersSet);
                    wordN += 1;
                    if (wordN % 100 == 0)
                    {
                        loggersSet.UserFriendlyLogger.LogInformation($"Calculate_ProjectionIndices_Variant3 iteration; WordN: {wordN} done. Energy: {energy}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
                        stopwatch.Restart();
                    }
                }                
            }

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("Calculate_ProjectionIndices_Variant3 totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }

        private float OptimizeWordProjection(Word word, int[] wordsProjectionIndices, NewVectorsAndMatrices newVectorsAndMatrices, Buffers buffers, ILoggersSet loggersSet)
        {
            //var stopwatch = Stopwatch.StartNew();            

            var dependentWords = newVectorsAndMatrices.Temp_DependentWords[word.Index];

            int minEnergyBitIndex = -1;
            float minEnergy = Single.MaxValue;
            for (int bitIndex = 0; bitIndex < NewVectorLength; bitIndex += 1)
            {
                wordsProjectionIndices[word.Index] = bitIndex;

                newVectorsAndMatrices.Calculate_Partial(dependentWords,
                    Words_RU,
                    wordsProjectionIndices,                    
                    loggersSet);

                float energy = GetEnergy(newVectorsAndMatrices,
                    buffers,
                    loggersSet);
                if (energy < minEnergy)
                {
                    minEnergy = energy;
                    minEnergyBitIndex = bitIndex;
                }
                buffers.EnergiesOfBits[bitIndex] = energy;
            }

            wordsProjectionIndices[word.Index] = minEnergyBitIndex;

            newVectorsAndMatrices.Calculate_Partial(dependentWords,
                    Words_RU,
                    wordsProjectionIndices,
                    loggersSet);

            //loggersSet.UserFriendlyLogger.LogInformation($"OptimizeWordProjection done. minEnergy: {minEnergy}; Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return minEnergy;
        }

        private float GetEnergy(NewVectorsAndMatrices newVectorsAndMatrices,
            Buffers buffers,
            ILoggersSet loggersSet)
        {
            var proxWordsNewMatrix = newVectorsAndMatrices.ProxWordsNewMatrix;
            var dispersionOfPairs = buffers.DispersionOfPairs;

            //var stopwatch = Stopwatch.StartNew();            

            Parallel.For(0, newVectorsAndMatrices.Temp_PairGroups.Length, pairGroupIndex =>
            //for (int pairGroupIndex = 0; pairGroupIndex < newVectorsAndMatrices.Temp_PairGroups.Length; pairGroupIndex += 1)
            {
                var pairGroup = newVectorsAndMatrices.Temp_PairGroups[pairGroupIndex];
                var newDotProducts = buffers.PairGroupsNewDotProducts[pairGroupIndex];                
                var length = pairGroup.Length;

                float sum = 0.0f;
                for (int i = 0; i < length; i += 1)
                {
                    float newDotProduct = proxWordsNewMatrix[pairGroup[i]];
                    newDotProducts[i] = newDotProduct;
                    sum += newDotProduct;
                }

                float target = sum / length;
                TensorPrimitives.Subtract(newDotProducts, target, newDotProducts);
                TensorPrimitives.Multiply(newDotProducts, newDotProducts, newDotProducts);
                dispersionOfPairs[pairGroupIndex] = TensorPrimitives.Sum(newDotProducts) / length;
            });

            var energy = TensorPrimitives.Sum(dispersionOfPairs) / dispersionOfPairs.Length;

            //loggersSet.UserFriendlyLogger.LogInformation($"GetEnergy done. Energy: {energy}; Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return energy;
        }

        private class Buffers
        {
            public Buffers(NewVectorsAndMatrices newVectorsAndMatrices) 
            {
                PairGroupsNewDotProducts = new float[newVectorsAndMatrices.Temp_PairGroups.Length][];
                for (int i = 0; i < PairGroupsNewDotProducts.Length; i++)
                {
                    PairGroupsNewDotProducts[i] = new float[newVectorsAndMatrices.Temp_PairGroups[i].Length];
                }
                DispersionOfPairs = new float[newVectorsAndMatrices.Temp_PairGroups.Length];
            }

            public readonly float[] EnergiesOfBits = new float[NewVectorLength];            

            public readonly float[][] PairGroupsNewDotProducts;

            public readonly float[] DispersionOfPairs;
        }
    }    
}



//int bias = lastPrimaryWord.Index * Words.Count;

//int? nearestPrimaryWordIndex = primaryWords.Select((w, i) => (w, i))
//    .Where(t => !t.w.Temp_Flag)
//    .Select(t => (t.w, t.i, ProxWordsOldMatrix[bias + t.w.Index]))
//    .OrderByDescending(t => t.Item3)
//    .Select(t => t.i)
//    .FirstOrDefault();
//if (nearestPrimaryWordIndex is null)
//    break;
//int clusterIndex = nearestPrimaryWordIndex.Value;


//lastPrimaryWord = primaryWords[clusterIndex];
//lastPrimaryWord.Temp_Flag = true;                                

//foreach (int clusterIndex in Enumerable.Range(0, primaryWords.Length))
//{
//    var stopwatch = Stopwatch.StartNew();

//    Word[] clusterWords = newVectorsAndMatrices.Temp_ClusterWords[clusterIndex];

//    float energy = 0.0f;
//    foreach (Word word in clusterWords)
//    {
//        energy = OptimizeWordProjection(word, wordsProjectionIndices, newVectorsAndMatrices, buffers, loggersSet);
//    }

//    //lastPrimaryWord = primaryWords[clusterIndex];
//    //lastPrimaryWord.Temp_Flag = true;

//    loggersSet.UserFriendlyLogger.LogInformation($"Initial iteration; Cluster {clusterIndex} done. Energy: {energy}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);

//    if (clusterIndex % 10 == 0)
//        SaveToFile_ProjectionIndices(ProjectionOptimization_Algorithm_Variant3, $"ProjectionOptimization{clusterIndex,4}.bin", _loggersSet);
//}

//Word[] primaryWords = clusterization_Algorithm.PrimaryWords!;
//foreach (var primaryWord in primaryWords)
//{
//    primaryWord.Temp_Flag = false;
//} 