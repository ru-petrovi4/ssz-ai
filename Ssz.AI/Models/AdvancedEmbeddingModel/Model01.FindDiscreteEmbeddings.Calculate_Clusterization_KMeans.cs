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
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public partial class Model01
{
    public void Calculate_Clusterization_AlgorithmData_KMeans(LanguageInfo languageInfo, ILoggersSet loggersSet)
    {
        var words = languageInfo.Words;

        var clusterization_AlgorithmData_KMeans = new Clusterization_AlgorithmData(languageInfo, name: "KMeans");
        clusterization_AlgorithmData_KMeans.GenerateOwnedData(Constants.ClustersCount);
        languageInfo.Clusterization_AlgorithmData = clusterization_AlgorithmData_KMeans;

        var totalStopwatch = Stopwatch.StartNew();

        Random r = new(43);

        var primaryWords_Random = new Word[Constants.ClustersCount];
        for (int index = 0; index < primaryWords_Random.Length; index += 1)
        {
            for (; ; )
            {
                var word = words[r.Next(words.Count)];
                if (word.Temp_Flag)
                    continue;

                primaryWords_Random[index] = word;
                word.Temp_Flag = true;
                break;
            }
        }

        int Q = 0;
        double delta_llh = 0;

        ClusterInfo[] clusterInfos = new ClusterInfo[Constants.ClustersCount];
        clusterization_AlgorithmData_KMeans.ClusterInfos = clusterInfos;
        for (int clusterIndex = 0; clusterIndex < clusterInfos.Length; clusterIndex += 1)
        {
            ClusterInfo wordClustrer = new()
            {
                CentroidOldVectorNormalized = new float[Constants.OldVectorLength],
            };
            Array.Copy(primaryWords_Random[clusterIndex].OldVectorNormalized, wordClustrer.CentroidOldVectorNormalized, Constants.OldVectorLength);
            clusterInfos[clusterIndex] = wordClustrer;
        }

        Array.Clear(clusterization_AlgorithmData_KMeans.ClusterIndices);

        while (TimeSpan.FromMilliseconds(totalStopwatch.ElapsedMilliseconds) < TimeSpan.FromHours(1))
        {
            var stopwatch = Stopwatch.StartNew();
            Q += 1;

            int[] newClusterIndices = new int[words.Count];

            #region ЕXPECTATION                

            Parallel.For(0, words.Count, wordIndex =>
            {
                Word word = words[wordIndex];
                var oldVectorNormalized = word.OldVectorNormalized;
                int nearestClusterIndex = -1;
                float energyMin = float.MaxValue;
                for (int clusterIndex = 0; clusterIndex < clusterInfos.Length; clusterIndex += 1)
                {
                    var wordCluster_CentroidOldVectorNormalized = clusterInfos[clusterIndex].CentroidOldVectorNormalized;
                    float energy = ModelHelper.GetEnergy(oldVectorNormalized, wordCluster_CentroidOldVectorNormalized);
                    if (energy < energyMin)
                    {
                        energyMin = energy;
                        nearestClusterIndex = clusterIndex;
                    }
                }
                clusterInfos[nearestClusterIndex].WordsCount += 1;
                newClusterIndices[wordIndex] = nearestClusterIndex;
            });

            #endregion

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("ЕXPECTATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
            stopwatch.Restart();

            if (newClusterIndices.SequenceEqual(clusterization_AlgorithmData_KMeans.ClusterIndices))
            {
                loggersSet.UserFriendlyLogger.LogInformation("newClusterIndices.SequenceEqual(clusterIndices)");
                break;
            }

            clusterization_AlgorithmData_KMeans.ClusterIndices = newClusterIndices;

            #region MAXIMIZATION   

            Parallel.For(0, clusterInfos.Length, clusterIndex =>
            {
                var wordCluster_CentroidOldVector = clusterInfos[clusterIndex].CentroidOldVectorNormalized;
                Array.Clear(wordCluster_CentroidOldVector);
            });

            Parallel.For(0, words.Count, wordIndex =>
            {
                Word word = words[wordIndex];
                var wordOldVectrorNormalized = word.OldVectorNormalized;
                var wordCluster_CentroidOldVector = clusterInfos[clusterization_AlgorithmData_KMeans.ClusterIndices[wordIndex]].CentroidOldVectorNormalized;
                lock (wordCluster_CentroidOldVector)
                {
                    TensorPrimitives.Add(wordCluster_CentroidOldVector, wordOldVectrorNormalized, wordCluster_CentroidOldVector);
                }
            });

            Parallel.For(0, clusterInfos.Length, clusterIndex =>
            {
                var wordCluster_CentroidOldVector = clusterInfos[clusterIndex].CentroidOldVectorNormalized;
                float norm = TensorPrimitives.Norm(wordCluster_CentroidOldVector);
                TensorPrimitives.Divide(wordCluster_CentroidOldVector, norm, wordCluster_CentroidOldVector);
            });

            #endregion

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("MAXIMIZATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        }        

        totalStopwatch.Stop();
        loggersSet.UserFriendlyLogger.LogInformation("CalculateAlgorithmData_KMeans.PrimaryWords totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
    }
}

//Word[] primaryWords_KMeans = clusterization_AlgorithmData_KMeans.PrimaryWords;

//        Parallel.For(0, wordClusters.Length, clusterIndex =>
//        {
//            var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVectorNormalized;

//int nearestWordIndex = -1;
//float nearestDotProduct = 0.0f;
//            for (int wordIndex = 0; wordIndex<words.Count; wordIndex += 1)
//            {
//                Word word = words[wordIndex];
//var oldVectror = word.OldVectorNormalized;

//float dotProduct = TensorPrimitives.Dot(oldVectror, wordCluster_CentroidOldVector);
//                if (dotProduct > nearestDotProduct)
//                {
//                    nearestDotProduct = dotProduct;
//                    nearestWordIndex = wordIndex;
//                }
//            }

//            primaryWords_KMeans[clusterIndex] = words[nearestWordIndex];
//        });

//Array.Clear(clusterization_AlgorithmData_KMeans.IsPrimaryWord);
//foreach (var primaryWord in primaryWords_KMeans)
//{
//    clusterization_AlgorithmData_KMeans.IsPrimaryWord[primaryWord.Index] = true;
//}