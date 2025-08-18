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
        public void Calculate_Clusterization_Algorithm_Classes(ILoggersSet loggersSet)
        {
            Clusterization_Algorithm_Classes.ClusterIndices = new int[Words_RU.Count];

            var totalStopwatch = Stopwatch.StartNew();

            var r = new Random();

            int Q = 0;
            const int Q_MAX = Int32.MaxValue;
            double delta_llh = 0;
            //float minDotProduct = 0.33f; // 1093 clusters
            float minDotProduct = 0.23f;

            List<WordCluster> wordClusters = new List<WordCluster>(PrimaryWordsCount);

            for (int wordIndex = 0; wordIndex < Words_RU.Count; wordIndex += 1)
            {
                Words_RU[wordIndex].Temp_Flag = false;
            }
            var words_RandomOrder = new List<Word>(Words_RU.Count);
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

            Array.Clear(Clusterization_Algorithm_Classes.ClusterIndices);

            while (TimeSpan.FromMilliseconds(totalStopwatch.ElapsedMilliseconds) < TimeSpan.FromHours(1))
            {
                var stopwatch = Stopwatch.StartNew();
                Q += 1;

                int[] newClusterIndices = new int[Words_RU.Count];

                #region ЕXPECTATION                

                for (int i = 0; i < words_RandomOrder.Count; i += 1)
                {
                    Word word = words_RandomOrder[i];
                    var oldVectror = word.OldVector;
                    //int indexBias = word.Index * Words.Count;

                    int nearestClusterIndex = -1;
                    float nearestDotProduct = 0.0f;
                    for (int clusterIndex = 0; clusterIndex < wordClusters.Count; clusterIndex += 1)
                    {
                        var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
                        float dotProduct = TensorPrimitives.Dot(oldVectror, wordCluster_CentroidOldVector);
                        //float dotProduct = ProxWordsMatrix[indexBias + wordClusters[clusterIndex].PrimaryWordIndex];

                        if (dotProduct > nearestDotProduct && dotProduct > minDotProduct)
                        {
                            nearestDotProduct = dotProduct;
                            nearestClusterIndex = clusterIndex;
                        }
                    }
                    if (nearestClusterIndex == -1)
                    {
                        //int nearestPrimaryWordIndex = -1;
                        //nearestDotProduct = 0.0f;
                        //for (int primaryWordIndex = 0; primaryWordIndex < Words.Count; primaryWordIndex += 1)
                        //{
                        //    Word primaryWord = Words[primaryWordIndex];
                        //    if (primaryWord.Temp_Flag) // In cluster already
                        //        continue;

                        //    //var oldVectror = primaryWord.OriginalNormalizedVectorArray;
                        //    //float dotProduct = TensorPrimitives.Dot(oldVectror, wordCluster_OriginalNormalizedVectorArray);
                        //    float dotProduct = ProxWordsMatrix[indexBias + primaryWordIndex];
                        //    if (dotProduct > nearestDotProduct && dotProduct > minDotProduct)
                        //    {
                        //        nearestDotProduct = dotProduct;
                        //        nearestPrimaryWordIndex = primaryWordIndex;
                        //    }                            
                        //}
                        //if (nearestPrimaryWordIndex == -1)
                        //    throw new Exception();
                        WordCluster wordCluster = new()
                        {
                            CentroidOldVector = (float[])word.OldVector.Clone(),
                            PrimaryWordIndex = word.Index,                            
                        };
                        nearestClusterIndex = wordClusters.Count;
                        wordClusters.Add(wordCluster);
                    }

                    wordClusters[nearestClusterIndex].WordsCount += 1;
                    newClusterIndices[word.Index] = nearestClusterIndex;
                }                

                #endregion

                stopwatch.Stop();
                loggersSet.UserFriendlyLogger.LogInformation("ЕXPECTATION done. wordClusters.Count=" + wordClusters.Count + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
                stopwatch.Restart();

                if (newClusterIndices.SequenceEqual(Clusterization_Algorithm_Classes.ClusterIndices))
                {
                    loggersSet.UserFriendlyLogger.LogInformation("newClusterIndices.SequenceEqual(clusterIndices)");
                    break;
                }

                Clusterization_Algorithm_Classes.ClusterIndices = newClusterIndices;

                #region MAXIMIZATION   

                Parallel.For(0, wordClusters.Count, clusterIndex =>
                {
                    var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
                    Array.Clear(wordCluster_CentroidOldVector);
                });

                Parallel.For(0, Words_RU.Count, wordIndex =>
                {
                    Word word = Words_RU[wordIndex];
                    var oldVectror = word.OldVector;
                    var wordCluster_CentroidOldVector = wordClusters[Clusterization_Algorithm_Classes.ClusterIndices[wordIndex]].CentroidOldVector;
                    lock (wordCluster_CentroidOldVector)
                    {
                        TensorPrimitives.Add(wordCluster_CentroidOldVector, word.OldVector, wordCluster_CentroidOldVector);
                    }
                });

                Parallel.For(0, wordClusters.Count, clusterIndex =>
                {
                    var wordCluster_CentroidOldVector = wordClusters[clusterIndex].CentroidOldVector;
                    float norm = TensorPrimitives.Norm(wordCluster_CentroidOldVector);
                    TensorPrimitives.Divide(wordCluster_CentroidOldVector, norm, wordCluster_CentroidOldVector);
                });                

                #endregion

                stopwatch.Stop();
                loggersSet.UserFriendlyLogger.LogInformation("MAXIMIZATION done. delta_llh=" + delta_llh + "; Q=" + Q + " Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
            }

            Word[] primaryWords_Classes = new Word[wordClusters.Count];
            Parallel.For(0, wordClusters.Count, clusterIndex =>
            {
                Word primaryWord = Clusterization_Algorithm_Classes.ClusterIndices.Select((ci, i) => (Words_RU[i], ci)).Where(it => it.Item2 == clusterIndex).MaxBy(it => it.Item1.Freq).Item1;
                primaryWords_Classes[clusterIndex] = primaryWord;
            });
            Clusterization_Algorithm_Classes.PrimaryWords = primaryWords_Classes;

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("CalculateAlgorithm_Classes.PrimaryWords totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }        
    }    
}

//public float[] DotProducts = null!;