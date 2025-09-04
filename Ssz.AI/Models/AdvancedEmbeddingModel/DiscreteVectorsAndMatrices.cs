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

public class DiscreteVectorsAndMatrices : ISerializableModelObject
{
    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] DiscreteVectors = null!;

    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] DiscreteVectors_PrimaryOnly = null!;

    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] DiscreteVectors_SecondaryOnly = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsDiscreteMatrix = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsDiscreteMatrix_PrimaryOnly = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsDiscreteMatrix_SecondaryOnly = null!;

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
    public bool[] Temp_ProxWordsDiscreteMatrix_InPairGroups = null!;

    public void GenerateOwnedData(int wordsCount)
    {
        DiscreteVectors = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            DiscreteVectors[wordIndex] = new float[Model01.Constants.DiscreteVectorLength];
        }

        DiscreteVectors_PrimaryOnly = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            DiscreteVectors_PrimaryOnly[wordIndex] = new float[Model01.Constants.DiscreteVectorLength];
        }

        DiscreteVectors_SecondaryOnly = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            DiscreteVectors_SecondaryOnly[wordIndex] = new float[Model01.Constants.DiscreteVectorLength];
        }

        ProxWordsDiscreteMatrix = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_PrimaryOnly = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_SecondaryOnly = new float[wordsCount * wordsCount];
    }

    public void Prepare(Clusterization_Algorithm clusterization_Algorithm, List<Word> words, MatrixFloat proxWordsOldMatrix)
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

        Temp_ProxWordsDiscreteMatrix_InPairGroups = new bool[wordsCount * wordsCount];
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
                Temp_ProxWordsDiscreteMatrix_InPairGroups[index] = true;
            }
            Temp_PairGroups[rangeIndex] = pairGroup;
            high = low;
        }
    } 
    
    /// <summary>
    ///     Calculates all vectors and matrices.
    /// </summary>
    /// <param name="words"></param>
    /// <param name="wordsProjectionIndices"></param>
    /// <param name="loggersSet"></param>
    public void Calculate_DiscreteVectorsAndMatrices(List<Word> words,
        int[] wordsProjectionIndices, 
        ILoggersSet loggersSet)
    {
        var wordsSubArray = words.ToArray();

        CalculateDiscreteVectorsOnly(wordsSubArray,
            wordsProjectionIndices,
            loggersSet);

        var stopwatch = Stopwatch.StartNew();

        ParallelLoopResult parallelLoopResult = Parallel.For(0, wordsSubArray.Length, i1 =>
        {
            int wordIndex1 = wordsSubArray[i1].Index;
            int indexBias = wordIndex1 * words.Count;

            var discreteVector = DiscreteVectors[wordIndex1];
            var discreteVector_PrimaryOnly = DiscreteVectors_PrimaryOnly[wordIndex1];
            var discreteVector_SecondaryOnly = DiscreteVectors_SecondaryOnly[wordIndex1];
            for (var i2 = 0; i2 < wordsSubArray.Length; i2 += 1)
            {
                int wordIndex2 = wordsSubArray[i2].Index;
                int matrixIndex = indexBias + wordIndex2;
                ProxWordsDiscreteMatrix[matrixIndex] = TensorPrimitives.Dot(discreteVector, DiscreteVectors[wordIndex2]);
                ProxWordsDiscreteMatrix_PrimaryOnly[matrixIndex] = TensorPrimitives.Dot(discreteVector_PrimaryOnly, DiscreteVectors_PrimaryOnly[wordIndex2]);
                ProxWordsDiscreteMatrix_SecondaryOnly[matrixIndex] = TensorPrimitives.Dot(discreteVector_SecondaryOnly, DiscreteVectors_SecondaryOnly[wordIndex2]);
            }
        });

        loggersSet.UserFriendlyLogger.LogInformation("Calculate_Full done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    /// <summary>
    ///     Calculates all vectors and partially matrices.
    /// </summary>
    /// <param name="wordsSubArray"></param>
    /// <param name="words"></param>
    /// <param name="wordsProjectionIndices"></param>
    /// <param name="loggersSet"></param>
    public void Calculate_DiscreteVectorsAndMatricesPartial(Word[] wordsSubArray,
        List<Word> words,
        int[] wordsProjectionIndices,
        ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        CalculateDiscreteVectorsOnly(wordsSubArray,
            wordsProjectionIndices,
            loggersSet);
        
        Parallel.For(0, words.Count, i1 =>
        {
            int wordIndex1 = words[i1].Index;
            int indexBias = wordIndex1 * words.Count;

            var discreteVector = DiscreteVectors[wordIndex1];
            //var discreteVector_PrimaryOnly = DiscreteVectors_PrimaryOnly[wordIndex1];
            //var discreteVector_SecondaryOnly = DiscreteVectors_SecondaryOnly[wordIndex1];
            for (var i2 = 0; i2 < wordsSubArray.Length; i2 += 1)
            {
                int wordIndex2 = wordsSubArray[i2].Index;
                int matrixIndex = indexBias + wordIndex2;
                if (Temp_ProxWordsDiscreteMatrix_InPairGroups[matrixIndex])
                    ProxWordsDiscreteMatrix[wordIndex2 * words.Count + wordIndex1] = 
                        ProxWordsDiscreteMatrix[matrixIndex] = 
                        TensorPrimitives.Dot(discreteVector, DiscreteVectors[wordIndex2]);
                //ProxWordsDiscreteMatrix_PrimaryOnly[matrixIndex] = TensorPrimitives.Dot(discreteVector_PrimaryOnly, DiscreteVectors_PrimaryOnly[wordIndex2]);
                //ProxWordsDiscreteMatrix_SecondaryOnly[matrixIndex] = TensorPrimitives.Dot(discreteVector_SecondaryOnly, DiscreteVectors_SecondaryOnly[wordIndex2]);
            }
        });            

        //loggersSet.UserFriendlyLogger.LogInformation("Calculate_Parital done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    /// <summary>
    ///     Calculaters vectors only.
    /// </summary>
    /// <param name="wordsSubArray"></param>
    /// <param name="wordsProjectionIndices"></param>
    /// <param name="loggersSet"></param>
    public void CalculateDiscreteVectorsOnly(Word[] wordsSubArray,            
        int[] wordsProjectionIndices,
        ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        foreach (Word word in wordsSubArray)
        {
            var discreteVector = DiscreteVectors[word.Index];
            var discreteVector_PrimaryOnly = DiscreteVectors_PrimaryOnly[word.Index];
            var discreteVector_SecondaryOnly = DiscreteVectors_SecondaryOnly[word.Index];
            Array.Clear(discreteVector);
            Array.Clear(discreteVector_PrimaryOnly);
            Array.Clear(discreteVector_SecondaryOnly);
            var temp_Top8ProxPrimaryWords = Temp_Top8ProxPrimaryWords[word.Index];
            var temp_Top8ProxWords = Temp_Top8ProxWords[word.Index];
            for (int i = 0; i < temp_Top8ProxPrimaryWords!.Length; i += 1)
            {
                int wordProjectionIndex = wordsProjectionIndices[temp_Top8ProxPrimaryWords[i].Item2.Index];
                if (wordProjectionIndex >= 0)
                {
                    discreteVector[wordProjectionIndex] = 1.0f;
                    discreteVector_PrimaryOnly[wordProjectionIndex] = 1.0f;
                }
            }
            for (int i = 0; i < temp_Top8ProxWords.Length; i += 1)
            {
                int wordProjectionIndex = wordsProjectionIndices[temp_Top8ProxWords[i].Item2.Index];
                if (wordProjectionIndex >= 0)
                {
                    discreteVector[wordProjectionIndex] = 1.0f;
                    discreteVector_SecondaryOnly[wordProjectionIndex] = 1.0f;
                }
            }
        }

        stopwatch.Stop();
        //loggersSet.UserFriendlyLogger.LogInformation("CalculateDiscreteVectors done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void SerializeOwnedData(SerializationWriter serializationWriter, object? context)
    {
        using (serializationWriter.EnterBlock(1))
        {
            serializationWriter.Write(DiscreteVectors.Length);
            if (DiscreteVectors is not null)
                foreach (var vectorNew in DiscreteVectors)
                {
                    serializationWriter.WriteArray(vectorNew);
                }
            //serializationWriter.WriteArray(algorithm.ProxWordsDiscreteMatrix);

            serializationWriter.Write(DiscreteVectors_PrimaryOnly.Length);
            if (DiscreteVectors_PrimaryOnly is not null)
                foreach (var vectorNew in DiscreteVectors_PrimaryOnly)
                {
                    serializationWriter.WriteArray(vectorNew);
                }
            //serializationWriter.WriteArray(algorithm.ProxWordsDiscreteMatrix_PrimaryOnly);

            serializationWriter.Write(DiscreteVectors_SecondaryOnly.Length);
            if (DiscreteVectors_SecondaryOnly is not null)
                foreach (var vectorNew in DiscreteVectors_SecondaryOnly)
                {
                    serializationWriter.WriteArray(vectorNew);
                }
        }
    }

    public void DeserializeOwnedData(SerializationReader serializationReader, object? context)
    {
        int wordsCount = 0;
        using (Block block = serializationReader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    int discreteVectorsLength = serializationReader.ReadInt32();
                    wordsCount = discreteVectorsLength;
                    if (discreteVectorsLength > 0)
                    {
                        var discreteVectors = new float[discreteVectorsLength][];
                        foreach (int i in Enumerable.Range(0, discreteVectorsLength))
                        {
                            discreteVectors[i] = serializationReader.ReadArray<float>()!;
                        }
                        DiscreteVectors = discreteVectors;
                    }
                    //algorithm.ProxWordsDiscreteMatrix = serializationReader.ReadArray<float>();

                    discreteVectorsLength = serializationReader.ReadInt32();
                    if (discreteVectorsLength > 0)
                    {
                        var discreteVectors_PrimaryOnly = new float[discreteVectorsLength][];
                        foreach (int i in Enumerable.Range(0, discreteVectorsLength))
                        {
                            discreteVectors_PrimaryOnly[i] = serializationReader.ReadArray<float>()!;
                        }
                        DiscreteVectors_PrimaryOnly = discreteVectors_PrimaryOnly;
                    }
                    //algorithm.ProxWordsDiscreteMatrix_PrimaryOnly = serializationReader.ReadArray<float>();

                    discreteVectorsLength = serializationReader.ReadInt32();
                    if (discreteVectorsLength > 0)
                    {
                        var discreteVectors_SecondaryOnly = new float[discreteVectorsLength][];
                        foreach (int i in Enumerable.Range(0, discreteVectorsLength))
                        {
                            discreteVectors_SecondaryOnly[i] = serializationReader.ReadArray<float>()!;
                        }
                        DiscreteVectors_SecondaryOnly = discreteVectors_SecondaryOnly;
                    }
                    //algorithm.ProxWordsDiscreteMatrix_SecondaryOnly = serializationReader.ReadArray<float>();
                    break;
            }
        }

        ProxWordsDiscreteMatrix = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_PrimaryOnly = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_SecondaryOnly = new float[wordsCount * wordsCount];
    }
}

// Word[] except_WordsSubArray = words.Except(wordsSubArray).ToArray();