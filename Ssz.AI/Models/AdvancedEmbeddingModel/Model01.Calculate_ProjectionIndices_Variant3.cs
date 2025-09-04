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
    public partial class Model01
    {
        public void Calculate_ProjectionIndices_Variant3(LanguageInfo languageInfo, ILoggersSet loggersSet)
        {
            Clusterization_AlgorithmData clusterization_AlgorithmData = languageInfo.Clusterization_AlgorithmData;
           
            var words = languageInfo.Words;
            
            var projectionOptimization_AlgorithmData_Variant3 = languageInfo.ProjectionOptimization_AlgorithmData;

            var totalStopwatch = Stopwatch.StartNew();

            var r = new Random();

            projectionOptimization_AlgorithmData_Variant3.WordsProjectionIndices = new int[words.Count];
            //LoadFromFile_ProjectionIndices(ProjectionOptimization_AlgorithmData_Variant3, "ProjectionOptimization.bin", _loggersSet);
            var wordsProjectionIndices = projectionOptimization_AlgorithmData_Variant3.WordsProjectionIndices;

            //Random initial hash
            foreach (int wordIndex in Enumerable.Range(0, wordsProjectionIndices.Length))
            {
                wordsProjectionIndices[wordIndex] = r.Next(Constants.DiscreteVectorLength);
            }                                      

            DiscreteVectorsAndMatrices discreteVectorsAndMatrices = new();
            discreteVectorsAndMatrices.GenerateOwnedData(words.Count);
            discreteVectorsAndMatrices.Prepare(clusterization_AlgorithmData, words, languageInfo.ProxWordsOldMatrix);
            discreteVectorsAndMatrices.Calculate_DiscreteVectorsAndMatrices(words, wordsProjectionIndices, loggersSet);

            Buffers buffers = new(discreteVectorsAndMatrices);                                              

            for (int i = 0; i < 1; i += 1)
            {
                var words_RandomOrder = new List<Word>(words.Count);
                foreach (var word in words)
                {
                    word.Temp_Flag = false;
                }
                for (int index = 0; index < words.Count; index += 1)
                {
                    for (; ; )
                    {
                        var word = words[r.Next(words.Count)];
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
                    energy = OptimizeWordProjection(languageInfo, word, discreteVectorsAndMatrices, buffers, loggersSet);
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

        private float OptimizeWordProjection(LanguageInfo languageInfo, Word word, DiscreteVectorsAndMatrices discreteVectorsAndMatrices, Buffers buffers, ILoggersSet loggersSet)
        {
            //var stopwatch = Stopwatch.StartNew();            

            var dependentWords = discreteVectorsAndMatrices.Temp_DependentWords[word.Index];

            int[] wordsProjectionIndices = languageInfo.ProjectionOptimization_AlgorithmData.WordsProjectionIndices;

            int minEnergyBitIndex = -1;
            float minEnergy = Single.MaxValue;
            for (int bitIndex = 0; bitIndex < Constants.DiscreteVectorLength; bitIndex += 1)
            {
                wordsProjectionIndices[word.Index] = bitIndex;

                discreteVectorsAndMatrices.Calculate_DiscreteVectorsAndMatricesPartial(dependentWords,
                    languageInfo.Words,
                    wordsProjectionIndices,                    
                    loggersSet);

                float energy = GetEnergy(discreteVectorsAndMatrices,
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

            discreteVectorsAndMatrices.Calculate_DiscreteVectorsAndMatricesPartial(dependentWords,
                    languageInfo.Words,
                    wordsProjectionIndices,
                    loggersSet);

            //loggersSet.UserFriendlyLogger.LogInformation($"OptimizeWordProjection done. minEnergy: {minEnergy}; Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return minEnergy;
        }

        private float GetEnergy(DiscreteVectorsAndMatrices discreteVectorsAndMatrices,
            Buffers buffers,
            ILoggersSet loggersSet)
        {
            var proxWordsDiscreteMatrix = discreteVectorsAndMatrices.ProxWordsDiscreteMatrix;
            var dispersionOfRange_Collection = buffers.DispersionOfRange_Collection;

            //var stopwatch = Stopwatch.StartNew();            

            Parallel.For(0, discreteVectorsAndMatrices.Temp_RangeDataIndicesCollection.Length, rangeIndex =>
            //for (int pairGroupIndex = 0; pairGroupIndex < discreteVectorsAndMatrices.Temp_PairGroups.Length; pairGroupIndex += 1)
            {
                var rangeDataIndices = discreteVectorsAndMatrices.Temp_RangeDataIndicesCollection[rangeIndex];
                var rangeDiscreteDotProductsCollection = buffers.RangeDiscreteDotProductsCollection[rangeIndex];                
                var dataIndicesCount = rangeDataIndices.Length;

                float sumOfDiscreteDotProducts = 0.0f;
                for (int i = 0; i < dataIndicesCount; i += 1)
                {
                    float discreteDotProduct = proxWordsDiscreteMatrix[rangeDataIndices[i]];
                    rangeDiscreteDotProductsCollection[i] = discreteDotProduct;
                    sumOfDiscreteDotProducts += discreteDotProduct;
                }

                float averageOfDiscreteDotProducts = sumOfDiscreteDotProducts / dataIndicesCount;
                TensorPrimitives.Subtract(rangeDiscreteDotProductsCollection, averageOfDiscreteDotProducts, rangeDiscreteDotProductsCollection);                
                dispersionOfRange_Collection[rangeIndex] = TensorPrimitives.SumOfSquares(rangeDiscreteDotProductsCollection) / dataIndicesCount;
            });

            var energy = TensorPrimitives.Sum(dispersionOfRange_Collection) / dispersionOfRange_Collection.Length;

            //loggersSet.UserFriendlyLogger.LogInformation($"GetEnergy done. Energy: {energy}; Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return energy;
        }

        private class Buffers
        {
            public Buffers(DiscreteVectorsAndMatrices discreteVectorsAndMatrices) 
            {
                RangeDiscreteDotProductsCollection = new float[discreteVectorsAndMatrices.Temp_RangeDataIndicesCollection.Length][];
                for (int i = 0; i < RangeDiscreteDotProductsCollection.Length; i++)
                {
                    RangeDiscreteDotProductsCollection[i] = new float[discreteVectorsAndMatrices.Temp_RangeDataIndicesCollection[i].Length];
                }
                DispersionOfRange_Collection = new float[discreteVectorsAndMatrices.Temp_RangeDataIndicesCollection.Length];
            }

            public readonly float[] EnergiesOfBits = new float[Constants.DiscreteVectorLength];

            /// <summary>
            ///     [Range of cosine similarity [0.50-0.55), [0.55-0.60), [0.95-1.00); [ProxWordsDisctreMatrix value]]
            /// </summary>
            public readonly float[][] RangeDiscreteDotProductsCollection;

            /// <summary>
            ///     [Range of cosine similarity [0.50-0.55), [0.55-0.60), [0.95-1.00); Dispersion]
            /// </summary>
            public readonly float[] DispersionOfRange_Collection;
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

//    Word[] clusterWords = discreteVectorsAndMatrices.Temp_ClusterWords[clusterIndex];

//    float energy = 0.0f;
//    foreach (Word word in clusterWords)
//    {
//        energy = OptimizeWordProjection(word, wordsProjectionIndices, discreteVectorsAndMatrices, buffers, loggersSet);
//    }

//    //lastPrimaryWord = primaryWords[clusterIndex];
//    //lastPrimaryWord.Temp_Flag = true;

//    loggersSet.UserFriendlyLogger.LogInformation($"Initial iteration; Cluster {clusterIndex} done. Energy: {energy}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);

//    if (clusterIndex % 10 == 0)
//        SaveToFile_ProjectionIndices(ProjectionOptimization_AlgorithmData_Variant3, $"ProjectionOptimization{clusterIndex,4}.bin", _loggersSet);
//}

//Word[] primaryWords = clusterization_AlgorithmData.PrimaryWords!;
//foreach (var primaryWord in primaryWords)
//{
//    primaryWord.Temp_Flag = false;
//} 