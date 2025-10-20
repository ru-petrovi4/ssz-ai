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
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model01
    {
        public ProjectionOptimization_AlgorithmData Calculate_ProjectionIndices_V4(
            List<Word> words, 
            List<ClusterInfo> clusterInfos,
            MatrixFloat wordsDistancesOldMatrix,
            Random r, 
            ILoggersSet loggersSet)
        {
            var projectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "V4");
            projectionOptimization_AlgorithmData.GenerateOwnedData(words.Count);            

            var totalStopwatch = Stopwatch.StartNew();
            
            int[] hashProjectionIndices = new int[clusterInfos.Count];
            foreach (int clusterIndex in Enumerable.Range(0, clusterInfos.Count))
            {
                hashProjectionIndices[clusterIndex] = clusterIndex;
            }
            r.Shuffle(hashProjectionIndices);
            foreach (int clusterIndex in Enumerable.Range(0, clusterInfos.Count))
            {
                clusterInfos[clusterIndex].HashProjectionIndex = hashProjectionIndices[clusterIndex];
            }

            //Random initial hash
            var wordsHashProjectionIndices = projectionOptimization_AlgorithmData.WordsHashProjectionIndices;
            foreach (int wordIndex in Enumerable.Range(0, wordsHashProjectionIndices.Length))
            {
                wordsHashProjectionIndices[wordIndex] = r.Next(Constants.DiscreteVectorLength);
            }

            DiscreteVectorsAndMatrices discreteVectorsAndMatrices = new();
            discreteVectorsAndMatrices.GenerateOwnedData(words.Count);
            discreteVectorsAndMatrices.Prepare(clusterInfos, words, wordsDistancesOldMatrix);
            discreteVectorsAndMatrices.Calculate_DiscreteVectorsAndMatrices(words, wordsHashProjectionIndices, loggersSet);

            Buffers buffers = new(discreteVectorsAndMatrices);

            // TEMPCODE
            for (int i = 0; i < 0; i += 1)
            {
                var words_RandomOrder = words.ToArray();
                r.Shuffle(words_RandomOrder);

                var stopwatch = Stopwatch.StartNew();
                float energy;
                int wordN = 0;
                foreach (Word word in words_RandomOrder)
                {
                    energy = OptimizeWordProjection_V4(
                        projectionOptimization_AlgorithmData,
                        words,
                        word,
                        discreteVectorsAndMatrices, 
                        buffers, 
                        loggersSet);
                    wordN += 1;
                    if (wordN % 100 == 0)
                    {
                        loggersSet.UserFriendlyLogger.LogInformation($"Calculate_ProjectionIndices_V4 iteration; WordN: {wordN} done. Energy: {energy}; Elapsed Milliseconds: " + stopwatch.ElapsedMilliseconds);
                        stopwatch.Restart();
                    }
                }
            }

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("Calculate_ProjectionIndices_V4 totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);

            return projectionOptimization_AlgorithmData;
        }

        private float OptimizeWordProjection_V4(
            ProjectionOptimization_AlgorithmData projectionOptimization_AlgorithmData,
            List<Word> words, 
            Word word, 
            DiscreteVectorsAndMatrices discreteVectorsAndMatrices, 
            Buffers buffers, 
            ILoggersSet loggersSet)
        {
            //var stopwatch = Stopwatch.StartNew();            

            var dependentWords = discreteVectorsAndMatrices.Temp_DependentWords[word.Index];

            int[] wordsProjectionIndices = projectionOptimization_AlgorithmData.WordsHashProjectionIndices;

            int minEnergyBitIndex = -1;
            float minEnergy = Single.MaxValue;
            for (int bitIndex = 0; bitIndex < Constants.DiscreteVectorLength; bitIndex += 1)
            {
                wordsProjectionIndices[word.Index] = bitIndex;

                discreteVectorsAndMatrices.Calculate_DiscreteVectorsAndMatricesPartial(dependentWords,
                    words,
                    wordsProjectionIndices,                    
                    loggersSet);

                float energy = GetEnergy_V4(discreteVectorsAndMatrices,
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

            discreteVectorsAndMatrices.Calculate_DiscreteVectorsAndMatricesPartial(
                    dependentWords,
                    words,
                    wordsProjectionIndices,
                    loggersSet);

            //loggersSet.UserFriendlyLogger.LogInformation($"OptimizeWordProjection done. minEnergy: {minEnergy}; Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return minEnergy;
        }

        private float GetEnergy_V4(DiscreteVectorsAndMatrices discreteVectorsAndMatrices,
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
