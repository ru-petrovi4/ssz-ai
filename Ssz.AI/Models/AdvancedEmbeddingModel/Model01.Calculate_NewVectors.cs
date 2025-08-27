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
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model01
    {
        public void Calculate_NewVector_ToDisplay(List<Word> words, Word word, int wordNum)
        {
            Clusterization_Algorithm? CurrentClusterization_Algorithm_ToDisplay = null;

            if (CurrentClusterization_Algorithm_ToDisplay?.PrimaryWords is not null &&
                CurrentNewVectorsAndMatrices_ToDisplay?.Temp_Top8ProxPrimaryWords is not null &&
                CurrentNewVectorsAndMatrices_ToDisplay?.Temp_Top8ProxWords is not null &&
                CurrentProjectionOptimization_Algorithm_ToDisplay is not null)
            {                
                Word[] topProxPrimaryWords = CurrentNewVectorsAndMatrices_ToDisplay.Temp_Top8ProxPrimaryWords[word.Index].Select(it => it.Item2).ToArray();
                Word[] topProxSecondaryWords = CurrentNewVectorsAndMatrices_ToDisplay.Temp_Top8ProxWords[word.Index].Select(it => it.Item2).ToArray();
                
                switch (wordNum)
                {
                    case 1:
                        for (int wordIndex = 0; wordIndex < words.Count; wordIndex += 1)
                        {
                            words[wordIndex].Point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.None;
                        }
                        for (int index = 0; index < CurrentClusterization_Algorithm_ToDisplay.PrimaryWords.Length; index += 1)
                        {
                            CurrentClusterization_Algorithm_ToDisplay.PrimaryWords[index].Point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.PrimaryPoint;
                        }
                        for (int index = 0; index < topProxSecondaryWords.Length; index += 1)
                        {
                            topProxSecondaryWords[index].Point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.SecondaryPoint_Selected1;
                        }
                        for (int index = 0; index < topProxPrimaryWords.Length; index += 1)
                        {
                            var point = topProxPrimaryWords[index].Point;
                            if (point.GroupId_ToDisplay == (int)PointGroupId_ToDisplay.SecondaryPoint_Selected1)
                                point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.PrimaryAndSecondaryPoint_Selected1;
                            else
                                point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.PrimaryPoint_Selected1;
                        }
                        word.Point.GroupId_ToDisplay = (int)PointGroupId_ToDisplay.MainPoint1;
                        break;
                    //case 2:
                    //    for (int index = 0; index < topProxPrimaryWords.Length; index += 1)
                    //    {
                    //        topProxPrimaryWords[index].Point.GroupId = (int)PointGroupId.PrimaryPoint_Selected2;
                    //    }
                    //    for (int index = 0; index < topProxSecondaryWords.Length; index += 1)
                    //    {
                    //        topProxSecondaryWords[index].Point.GroupId = (int)PointGroupId.SecondaryPoint_Selected2;
                    //    }
                    //    break;
                }                

                var newVector =  new float[NewVectorLength];
                for (int i = 0; i < topProxPrimaryWords.Length; i += 1)
                {
                    newVector[CurrentProjectionOptimization_Algorithm_ToDisplay.WordsProjectionIndices[topProxPrimaryWords[i].Index]] = 3.0f;
                }
                for (int i = 0; i < topProxSecondaryWords.Length; i += 1)
                {
                    newVector[CurrentProjectionOptimization_Algorithm_ToDisplay.WordsProjectionIndices[topProxSecondaryWords[i].Index]] += 1.0f;
                }

                word.NewVector_ToDisplay = newVector;                             

                CreateCortexCopy();
            }
        }

        //public NewVectorsAndMatrices Calculate_NewVectors_NoClusters(Clusterization_Algorithm clusterization_Algorithm,
        //    ProjectionOptimization_Algorithm projectionOptimization_Algorithm,
        //    ILoggersSet loggersSet)
        //{
        //    var stopwatch = Stopwatch.StartNew();

        //    var newVectors = new float[Words.Count][];
        //    newVectors[0] = new float[0];
        //    var newVectors_PrimaryOnly = new float[Words.Count][];
        //    newVectors_PrimaryOnly[0] = new float[0];
        //    var newVectors_SecondaryOnly = new float[Words.Count][];
        //    newVectors_SecondaryOnly[0] = new float[0];

        //    ParallelLoopResult parallelLoopResult = Parallel.For(0, Words.Count, wordIndex =>
        //    {
        //        Word word = Words[wordIndex];
        //        int indexBias = word.Index * Words.Count;
        //        Word[] topProxPrimaryWords = clusterization_Algorithm.PrimaryWords!
        //            .OrderByDescending(x => ProxWordsOldMatrix[indexBias + x.Index])
        //            .Take(PrimaryWords_NewVector_BitsCount).ToArray();
        //        Word[] topProxSecondaryWords = Words
        //            .OrderByDescending(x => ProxWordsOldMatrix[indexBias + x.Index])
        //            .Take(SecondaryWords_NewVector_BitsCount).ToArray();

        //        var newVector = new float[NewVectorLength];
        //        for (int i = 0; i < topProxPrimaryWords.Length; i += 1)
        //        {
        //            newVector[projectionOptimization_Algorithm.WordsProjectionIndices[topProxPrimaryWords[i].Index]] = 1.0f;
        //        }
        //        for (int i = 0; i < topProxSecondaryWords.Length; i += 1)
        //        {
        //            newVector[projectionOptimization_Algorithm.WordsProjectionIndices[topProxSecondaryWords[i].Index]] = 1.0f;
        //        }
        //        newVectors[wordIndex] = newVector;

        //        var newVector_PrimaryOnly = new float[NewVectorLength];
        //        for (int i = 0; i < topProxPrimaryWords.Length; i += 1)
        //        {
        //            newVector_PrimaryOnly[projectionOptimization_Algorithm.WordsProjectionIndices[topProxPrimaryWords[i].Index]] = 1.0f;
        //        }                
        //        newVectors_PrimaryOnly[wordIndex] = newVector_PrimaryOnly;

        //        var newVector_SecondaryOnly = new float[NewVectorLength];                
        //        for (int i = 0; i < topProxSecondaryWords.Length; i += 1)
        //        {
        //            newVector_SecondaryOnly[projectionOptimization_Algorithm.WordsProjectionIndices[topProxSecondaryWords[i].Index]] = 1.0f;
        //        }
        //        newVectors_SecondaryOnly[wordIndex] = newVector_SecondaryOnly;
        //    });

        //    NewVectorsAndMatrices result = new();
        //    result.NewVectors = newVectors.ToList();
        //    result.NewVectors_PrimaryOnly = newVectors_PrimaryOnly.ToList();
        //    result.NewVectors_SecondaryOnly = newVectors_SecondaryOnly.ToList();

        //    stopwatch.Stop();
        //    loggersSet.UserFriendlyLogger.LogInformation(clusterization_Algorithm.Name + " CalculateNewVectors done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

        //    return result;
        //}

        public NewVectorsAndMatrices Calculate_NewVectorsAndMatrices(LanguageInfo languageInfo, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();            

            NewVectorsAndMatrices result = new();
            result.Initialize(languageInfo.Words.Count);
            result.InitializeTemp(languageInfo.Clusterization_Algorithm, languageInfo.Words, languageInfo.ProxWordsOldMatrix);
            result.Calculate_Full(languageInfo.Words, languageInfo.ProjectionOptimization_Algorithm.WordsProjectionIndices, loggersSet);

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("Clusterization:" + languageInfo.Clusterization_Algorithm.Name + "; ProjectionOptimization:" + languageInfo.ProjectionOptimization_Algorithm.Name + " CalculateNewVectors done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return result;
        }

        public NewVectorsAndMatrices Calculate_NewVectors(LanguageInfo languageInfo, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            NewVectorsAndMatrices result = new();
            result.Initialize(languageInfo.Words.Count);
            result.InitializeTemp(languageInfo.Clusterization_Algorithm, languageInfo.Words, languageInfo.ProxWordsOldMatrix);
            result.CalculateNewVectors(languageInfo.Words.ToArray(), languageInfo.ProjectionOptimization_Algorithm.WordsProjectionIndices, loggersSet);

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("ProjectionOptimization:" + languageInfo.ProjectionOptimization_Algorithm.Name + " CalculateNewVectors done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

            return result;
        }
    }    
}

///// <summary>
/////     [WordIndex1, WordIndex2] New words correlation matrix.
///// </summary>
//public float[] NewProxWordsMatrix2 = null!;

//||
//               primaryWordsCount != PrimaryWordsCount ||
//               primaryWords_FinalVector_BitsCount != PrimaryWords_FinalVector_BitsCount ||
//               secondaryWords_FinalVector_BitsCount != SecondaryWords_FinalVector_BitsCount


//PrimaryWordsCount = primaryWordsCount;
//PrimaryWords_FinalVector_BitsCount = primaryWords_FinalVector_BitsCount;
//SecondaryWords_FinalVector_BitsCount = secondaryWords_FinalVector_BitsCount;




