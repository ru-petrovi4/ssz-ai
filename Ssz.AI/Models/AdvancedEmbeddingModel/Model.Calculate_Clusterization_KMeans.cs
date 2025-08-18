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
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model
    {
        public void Calculate_Clusterization_Algorithm_KMeans(ILoggersSet loggersSet)
        {
            Clusterization_Algorithm_KMeans.ClusterIndices = new int[Words_RU.Count];

            var totalStopwatch = Stopwatch.StartNew();

            Random r = new();

            var primaryWords_Random = new Word[PrimaryWordsCount];
            for (int index = 0; index < primaryWords_Random.Length; index += 1)
            {
                for (; ; )
                {
                    var word = Words_RU[r.Next(Words_RU.Count)];
                    if (word.Temp_Flag)
                        continue;

                    primaryWords_Random[index] = word;
                    word.Temp_Flag = true;
                    break;
                }
            }

            int Q = 0;
            const int Q_MAX = Int32.MaxValue;
            double delta_llh = 0;
            
            WordCluster[] wordClusters = new WordCluster[PrimaryWordsCount];
            for (int clusterIndex = 0; clusterIndex < wordClusters.Length; clusterIndex += 1)
            {
                WordCluster wordClustrer = new()
                {
                    CentroidOldVector = new float[OldVectorLength],                    
                };
                Array.Copy(primaryWords_Random[clusterIndex].OldVector, wordClustrer.CentroidOldVector, OldVectorLength);
                wordClusters[clusterIndex] = wordClustrer;
            }

            Array.Clear(Clusterization_Algorithm_KMeans.ClusterIndices);

            while (TimeSpan.FromMilliseconds(totalStopwatch.ElapsedMilliseconds) < TimeSpan.FromHours(1))
            {
                var stopwatch = Stopwatch.StartNew();
                Q += 1;

                int[] newClusterIndices = new int[Words_RU.Count];

                #region ЕXPECTATION                

                Parallel.For(0, Words_RU.Count, wordIndex =>
                {
                    Word word = Words_RU[wordIndex];
                    var oldVectror = word.OldVector;
                    int nearestClusterIndex = -1;
                    float nearestDotProduct = 0.0f;
                    for (int clusterIndex = 0; clusterIndex < wordClusters.Length; clusterIndex += 1)
                    {
                        var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
                        float dotProduct = TensorPrimitives.Dot(oldVectror, wordCluster_CentroidOldVector);
                        if (dotProduct > nearestDotProduct) 
                        {
                            nearestDotProduct = dotProduct;
                            nearestClusterIndex = clusterIndex;
                        }
                    }
                    wordClusters[nearestClusterIndex].WordsCount += 1;
                    newClusterIndices[wordIndex] = nearestClusterIndex;
                });                

                #endregion

                stopwatch.Stop();
                loggersSet.UserFriendlyLogger.LogInformation("ЕXPECTATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
                stopwatch.Restart();

                if (newClusterIndices.SequenceEqual(Clusterization_Algorithm_KMeans.ClusterIndices))
                {
                    loggersSet.UserFriendlyLogger.LogInformation("newClusterIndices.SequenceEqual(clusterIndices)");
                    break;
                }

                Clusterization_Algorithm_KMeans.ClusterIndices = newClusterIndices;

                #region MAXIMIZATION   

                Parallel.For(0, wordClusters.Length, clusterIndex =>
                {
                    var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
                    Array.Clear(wordCluster_CentroidOldVector);
                });

                Parallel.For(0, Words_RU.Count, wordIndex =>
                {
                    Word word = Words_RU[wordIndex];
                    var oldVectror = word.OldVector;
                    var wordCluster_CentroidOldVector = wordClusters[Clusterization_Algorithm_KMeans.ClusterIndices[wordIndex]].CentroidOldVector; 
                    lock (wordCluster_CentroidOldVector) 
                    {
                        TensorPrimitives.Add(wordCluster_CentroidOldVector, word.OldVector, wordCluster_CentroidOldVector);
                    }                    
                });

                Parallel.For(0, wordClusters.Length, clusterIndex =>
                {
                    var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
                    float norm = TensorPrimitives.Norm(wordCluster_CentroidOldVector);
                    TensorPrimitives.Divide(wordCluster_CentroidOldVector, norm, wordCluster_CentroidOldVector);
                });

                #endregion

                stopwatch.Stop();
                loggersSet.UserFriendlyLogger.LogInformation("MAXIMIZATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
            }

            Word[] primaryWords_KMeans = new Word[PrimaryWordsCount];

            Parallel.For(0, wordClusters.Length, clusterIndex =>
            {
                var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;

                int nearestWordIndex = -1;
                float nearestDotProduct = 0.0f;
                for (int wordIndex = 0; wordIndex < Words_RU.Count; wordIndex += 1)
                {
                    Word word = Words_RU[wordIndex];
                    var oldVectror = word.OldVector;

                    float dotProduct = TensorPrimitives.Dot(oldVectror, wordCluster_CentroidOldVector);
                    if (dotProduct > nearestDotProduct)
                    {
                        nearestDotProduct = dotProduct;
                        nearestWordIndex = wordIndex;
                    }
                }

                primaryWords_KMeans[clusterIndex] = Words_RU[nearestWordIndex];
            });

            Clusterization_Algorithm_KMeans.PrimaryWords = primaryWords_KMeans;

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("CalculateAlgorithm_KMeans.PrimaryWords totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }
    }    
}