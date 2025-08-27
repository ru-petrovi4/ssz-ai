using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class NewVectorsAndMatrices : IOwnedDataSerializable
{
    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] NewVectors = null!;

    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] NewVectors_PrimaryOnly = null!;

    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] NewVectors_SecondaryOnly = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsNewMatrix = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsNewMatrix_PrimaryOnly = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsNewMatrix_SecondaryOnly = null!;

    /// <summary>
    ///     Top 8 word refs (ordered by proximity, nearest first) for each word.
    ///     (Proximity, Word)
    /// </summary>
    public (float, Word)[][] Temp_Top8ProxWords = null!;

    /// <summary>
    ///     Top 8 primary word refs (ordered by proximity, nearest first) for each word.
    ///     (Proximity, Word)
    /// </summary>
    public (float, Word)[][] Temp_Top8ProxPrimaryWords = null!;

    public Word[][] Temp_DependentWords = null!;        

    /// <summary>
    ///     [Clusters, Words]
    /// </summary>
    public Word[][] Temp_ClusterWords = null!;

    /// <summary>
    ///     [Range, ProxWordsOldMatrix Indices]
    /// </summary>
    public int[][] Temp_PairGroups = null!;

    /// <summary>
    /// 
    /// </summary>
    public bool[] Temp_ProxWordsNewMatrix_InPairGroups = null!;

    public void Initialize(int wordsCount)
    {
        NewVectors = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            NewVectors[wordIndex] = new float[Model01.NewVectorLength];
        }

        NewVectors_PrimaryOnly = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            NewVectors_PrimaryOnly[wordIndex] = new float[Model01.NewVectorLength];
        }

        NewVectors_SecondaryOnly = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            NewVectors_SecondaryOnly[wordIndex] = new float[Model01.NewVectorLength];
        }

        ProxWordsNewMatrix = new float[wordsCount * wordsCount];
        ProxWordsNewMatrix_PrimaryOnly = new float[wordsCount * wordsCount];
        ProxWordsNewMatrix_SecondaryOnly = new float[wordsCount * wordsCount];
    }

    public void InitializeTemp(Clusterization_Algorithm clusterization_Algorithm, List<Word> words, MatrixFloat proxWordsOldMatrix)
    {
        int wordsCount = words.Count;
        Word[] primaryWords = clusterization_Algorithm.PrimaryWords!;

        Temp_Top8ProxWords = new (float, Word)[wordsCount][];
        Temp_Top8ProxPrimaryWords = new (float, Word)[wordsCount][];
        Temp_ClusterWords = new Word[primaryWords.Length][];  
        
        List<Word>[] dependentWords = new List<Word>[wordsCount];
        foreach (int wordIndex in Enumerable.Range(0, dependentWords.Length))
        {
            dependentWords[wordIndex] = new List<Word>();
        }

        foreach (int clusterIndex in Enumerable.Range(0, Temp_ClusterWords.Length))
        {
            Word[] clusterWords = clusterization_Algorithm.ClusterIndices!
                .Select((ci, wordIndex) => (words[wordIndex], ci))
                .Where(it => it.Item2 == clusterIndex)
                .Select(it => it.Item1)
                .ToArray();
            Temp_ClusterWords[clusterIndex] = clusterWords;

            foreach (Word word in clusterWords)
            {
                int indexBias = word.Index * wordsCount;
                var temp_Top8ProxWords = clusterWords.Select(w => (proxWordsOldMatrix.Data[indexBias + w.Index], w))
                    .OrderByDescending(it => it.Item1)
                    .Take(8)
                    .ToArray();
                Temp_Top8ProxWords[word.Index] = temp_Top8ProxWords;
                foreach (var it in temp_Top8ProxWords)
                {
                    dependentWords[it.Item2.Index].Add(word);
                }

                var temp_Top8ProxPrimaryWords = primaryWords.Select(w => (proxWordsOldMatrix.Data[indexBias + w.Index], w))
                    .OrderByDescending(it => it.Item1)
                    .Take(8)
                    .ToArray();
                Temp_Top8ProxPrimaryWords[word.Index] = temp_Top8ProxPrimaryWords;
                foreach (var it in temp_Top8ProxPrimaryWords)
                {
                    dependentWords[it.Item2.Index].Add(word);
                }
            }                
        }

        Temp_ProxWordsNewMatrix_InPairGroups = new bool[wordsCount * wordsCount];
        Temp_DependentWords = new Word[dependentWords.Length][];  
        foreach (int wordIndex in Enumerable.Range(0, dependentWords.Length))
        {
            Temp_DependentWords[wordIndex] = dependentWords[wordIndex].Distinct().ToArray();
        }            
        float delta = 0.05f;
        int count = (int)((1.0f - 0.5f) / delta);
        Temp_PairGroups = new int[count][];
        float high = 1.0f;
        for (int rangeIndex = count - 1; rangeIndex >= 0; rangeIndex -= 1)
        {
            float low = high - delta;
            var pairGroup = proxWordsOldMatrix.Data.Select((f, i) => (f, i))
                .Where(it => low <= it.f && it.f < high)
                .Select(it => it.i)
                .ToArray();
            foreach (int index in pairGroup)
            {
                Temp_ProxWordsNewMatrix_InPairGroups[index] = true;
            }
            Temp_PairGroups[rangeIndex] = pairGroup;
            high = low;
        }
    } 
    
    public void Calculate_Full(List<Word> words,
        int[] wordsProjectionIndices, 
        ILoggersSet loggersSet)
    {
        var wordsSubArray = words.ToArray();

        CalculateNewVectors(wordsSubArray,
            wordsProjectionIndices,
            loggersSet);

        var stopwatch = Stopwatch.StartNew();

        ParallelLoopResult parallelLoopResult = Parallel.For(0, wordsSubArray.Length, i1 =>
        {
            int wordIndex1 = wordsSubArray[i1].Index;
            int indexBias = wordIndex1 * words.Count;

            var newVector = NewVectors[wordIndex1];
            var newVector_PrimaryOnly = NewVectors_PrimaryOnly[wordIndex1];
            var newVector_SecondaryOnly = NewVectors_SecondaryOnly[wordIndex1];
            for (var i2 = 0; i2 < wordsSubArray.Length; i2 += 1)
            {
                int wordIndex2 = wordsSubArray[i2].Index;
                int matrixIndex = indexBias + wordIndex2;
                ProxWordsNewMatrix[matrixIndex] = TensorPrimitives.Dot(newVector, NewVectors[wordIndex2]);
                ProxWordsNewMatrix_PrimaryOnly[matrixIndex] = TensorPrimitives.Dot(newVector_PrimaryOnly, NewVectors_PrimaryOnly[wordIndex2]);
                ProxWordsNewMatrix_SecondaryOnly[matrixIndex] = TensorPrimitives.Dot(newVector_SecondaryOnly, NewVectors_SecondaryOnly[wordIndex2]);
            }
        });

        loggersSet.UserFriendlyLogger.LogInformation("Calculate_Full done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void Calculate_Partial(Word[] wordsSubArray,
        List<Word> words,
        int[] wordsProjectionIndices,
        ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        CalculateNewVectors(wordsSubArray,
            wordsProjectionIndices,
            loggersSet);
        
        Parallel.For(0, words.Count, i1 =>
        {
            int wordIndex1 = words[i1].Index;
            int indexBias = wordIndex1 * words.Count;

            var newVector = NewVectors[wordIndex1];
            //var newVector_PrimaryOnly = NewVectors_PrimaryOnly[wordIndex1];
            //var newVector_SecondaryOnly = NewVectors_SecondaryOnly[wordIndex1];
            for (var i2 = 0; i2 < wordsSubArray.Length; i2 += 1)
            {
                int wordIndex2 = wordsSubArray[i2].Index;
                int matrixIndex = indexBias + wordIndex2;
                if (Temp_ProxWordsNewMatrix_InPairGroups[matrixIndex])
                    ProxWordsNewMatrix[wordIndex2 * words.Count + wordIndex1] = 
                        ProxWordsNewMatrix[matrixIndex] = 
                        TensorPrimitives.Dot(newVector, NewVectors[wordIndex2]);
                //ProxWordsNewMatrix_PrimaryOnly[matrixIndex] = TensorPrimitives.Dot(newVector_PrimaryOnly, NewVectors_PrimaryOnly[wordIndex2]);
                //ProxWordsNewMatrix_SecondaryOnly[matrixIndex] = TensorPrimitives.Dot(newVector_SecondaryOnly, NewVectors_SecondaryOnly[wordIndex2]);
            }
        });            

        //loggersSet.UserFriendlyLogger.LogInformation("Calculate_Parital done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void CalculateNewVectors(Word[] wordsSubArray,            
        int[] wordsProjectionIndices,
        ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        foreach (Word word in wordsSubArray)
        {
            var newVector = NewVectors[word.Index];
            var newVector_PrimaryOnly = NewVectors_PrimaryOnly[word.Index];
            var newVector_SecondaryOnly = NewVectors_SecondaryOnly[word.Index];
            Array.Clear(newVector);
            Array.Clear(newVector_PrimaryOnly);
            Array.Clear(newVector_SecondaryOnly);
            var temp_Top8ProxPrimaryWords = Temp_Top8ProxPrimaryWords[word.Index];
            var temp_Top8ProxWords = Temp_Top8ProxWords[word.Index];
            for (int i = 0; i < temp_Top8ProxPrimaryWords!.Length; i += 1)
            {
                int wordProjectionIndex = wordsProjectionIndices[temp_Top8ProxPrimaryWords[i].Item2.Index];
                if (wordProjectionIndex >= 0)
                {
                    newVector[wordProjectionIndex] = 1.0f;
                    newVector_PrimaryOnly[wordProjectionIndex] = 1.0f;
                }
            }
            for (int i = 0; i < temp_Top8ProxWords.Length; i += 1)
            {
                int wordProjectionIndex = wordsProjectionIndices[temp_Top8ProxWords[i].Item2.Index];
                if (wordProjectionIndex >= 0)
                {
                    newVector[wordProjectionIndex] = 1.0f;
                    newVector_SecondaryOnly[wordProjectionIndex] = 1.0f;
                }
            }
        }

        stopwatch.Stop();
        //loggersSet.UserFriendlyLogger.LogInformation("CalculateNewVectors done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void SerializeOwnedData(SerializationWriter serializationWriter, object? context)
    {
        using (serializationWriter.EnterBlock(1))
        {
            serializationWriter.Write(NewVectors.Length);
            if (NewVectors is not null)
                foreach (var vectorNew in NewVectors)
                {
                    serializationWriter.WriteArray(vectorNew);
                }
            //serializationWriter.WriteArray(algorithm.ProxWordsNewMatrix);

            serializationWriter.Write(NewVectors_PrimaryOnly.Length);
            if (NewVectors_PrimaryOnly is not null)
                foreach (var vectorNew in NewVectors_PrimaryOnly)
                {
                    serializationWriter.WriteArray(vectorNew);
                }
            //serializationWriter.WriteArray(algorithm.ProxWordsNewMatrix_PrimaryOnly);

            serializationWriter.Write(NewVectors_SecondaryOnly.Length);
            if (NewVectors_SecondaryOnly is not null)
                foreach (var vectorNew in NewVectors_SecondaryOnly)
                {
                    serializationWriter.WriteArray(vectorNew);
                }
        }
    }

    public void DeserializeOwnedData(SerializationReader serializationReader, object? context)
    {
        using (Block block = serializationReader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    int newVectorsLength = serializationReader.ReadInt32();
                    if (newVectorsLength > 0)
                    {
                        var newVectors = new float[newVectorsLength][];
                        foreach (int i in Enumerable.Range(0, newVectorsLength))
                        {
                            newVectors[i] = serializationReader.ReadArray<float>()!;
                        }
                        NewVectors = newVectors;
                    }
                    //algorithm.ProxWordsNewMatrix = serializationReader.ReadArray<float>();

                    newVectorsLength = serializationReader.ReadInt32();
                    if (newVectorsLength > 0)
                    {
                        var newVectors_PrimaryOnly = new float[newVectorsLength][];
                        foreach (int i in Enumerable.Range(0, newVectorsLength))
                        {
                            newVectors_PrimaryOnly[i] = serializationReader.ReadArray<float>()!;
                        }
                        NewVectors_PrimaryOnly = newVectors_PrimaryOnly;
                    }
                    //algorithm.ProxWordsNewMatrix_PrimaryOnly = serializationReader.ReadArray<float>();

                    newVectorsLength = serializationReader.ReadInt32();
                    if (newVectorsLength > 0)
                    {
                        var newVectors_SecondaryOnly = new float[newVectorsLength][];
                        foreach (int i in Enumerable.Range(0, newVectorsLength))
                        {
                            newVectors_SecondaryOnly[i] = serializationReader.ReadArray<float>()!;
                        }
                        NewVectors_SecondaryOnly = newVectors_SecondaryOnly;
                    }
                    //algorithm.ProxWordsNewMatrix_SecondaryOnly = serializationReader.ReadArray<float>();
                    break;
            }
        }
    }
}

// Word[] except_WordsSubArray = words.Except(wordsSubArray).ToArray();