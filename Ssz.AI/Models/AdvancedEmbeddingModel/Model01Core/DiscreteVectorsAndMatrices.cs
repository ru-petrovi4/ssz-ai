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

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class DiscreteVectorsAndMatrices : ISerializableModelObject
{
    /// <summary>
    ///     New vectors for each word.
    /// </summary>
    public float[][] DiscreteVectors = null!;

    /// <summary>
    ///     New vectors for each word with primary bits only.
    /// </summary>
    public float[][] DiscreteVectors_PrimaryBitsOnly = null!;

    /// <summary>
    ///     New vectors for each  word with secondary bits only.
    /// </summary>
    public float[][] DiscreteVectors_SecondaryBitsOnly = null!;

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
    public float[] ProxWordsDiscreteMatrix_PrimaryBitsOnly = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     New vectors scalar product. Each element - common bits count.
    /// </remarks>
    public float[] ProxWordsDiscreteMatrix_SecondaryBitsOnly = null!;

    /// <summary>
    ///     Top 8 word refs (ordered by proximity, nearest first) for each word.
    ///     (Proximity, Word)
    /// </summary>
    public (float, Word)[][] Temp_Top8ProxWords = null!;

    /// <summary>
    ///     Top 8 ClusterInfo refs (ordered by proximity, nearest first) for each word.
    ///     (Proximity, ClusterInfo)
    /// </summary>
    public (float, ClusterInfo)[][] Temp_Top8ProxClusterInfos = null!;

    public Word[][] Temp_DependentWords = null!;  

    /// <summary>
    ///     [Range of cosine similarity [0.50-0.55), [0.55-0.60), [0.95-1.00); [ProxWordsOldMatrix.Data Index]]
    /// </summary>
    public int[][] Temp_RangeDataIndicesCollection = null!;

    /// <summary>
    ///     ProxWordsOldMatrix.Data Indices which is in <see cref="Temp_RangeDataIndicesCollection" />
    /// </summary>
    public DenseMatrix<bool> Temp_InRangeDataIndicesCollection = null!;

    public void GenerateOwnedData(int wordsCount)
    {
        DiscreteVectors = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            DiscreteVectors[wordIndex] = new float[Model01.Constants.DiscreteVectorLength];
        }

        DiscreteVectors_PrimaryBitsOnly = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            DiscreteVectors_PrimaryBitsOnly[wordIndex] = new float[Model01.Constants.DiscreteVectorLength];
        }

        DiscreteVectors_SecondaryBitsOnly = new float[wordsCount][];            
        foreach (int wordIndex in Enumerable.Range(0, wordsCount))
        {
            DiscreteVectors_SecondaryBitsOnly[wordIndex] = new float[Model01.Constants.DiscreteVectorLength];
        }

        ProxWordsDiscreteMatrix = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_PrimaryBitsOnly = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_SecondaryBitsOnly = new float[wordsCount * wordsCount];
    }

    public void Prepare(Clusterization_AlgorithmData clusterization_AlgorithmData, List<Word> words, MatrixFloat_ColumnMajor proxWordsOldMatrix)
    {
        int wordsCount = words.Count;
        ClusterInfo[] clusterInfos = clusterization_AlgorithmData.ClusterInfos;

        Temp_Top8ProxWords = new (float, Word)[wordsCount][];
        Temp_Top8ProxClusterInfos = new (float, ClusterInfo)[wordsCount][];        
        
        List<Word>[] dependentWords = Enumerable.Range(0, wordsCount).Select(_ => new List<Word>()).ToArray();        

        foreach (int clusterIndex in Enumerable.Range(0, clusterInfos.Length))
        {
            Word[] clusterWords = clusterization_AlgorithmData.ClusterIndices
                .Select((ci, wordIndex) => (words[wordIndex], ci))
                .Where(it => it.Item2 == clusterIndex)
                .Select(it => it.Item1)
                .ToArray();            

            foreach (Word clusterWord in clusterWords)
            {
                int indexBias = clusterWord.Index * wordsCount;
                var temp_Top8ProxWords = clusterWords.Select(w => (proxWordsOldMatrix.Data[indexBias + w.Index], w))
                    .OrderByDescending(it => it.Item1)
                    .Take(8)
                    .ToArray();
                Temp_Top8ProxWords[clusterWord.Index] = temp_Top8ProxWords;
                foreach (var it in temp_Top8ProxWords)
                {
                    dependentWords[it.Item2.Index].Add(clusterWord);
                }

                var temp_Top8ProxClusterInfos = clusterInfos.Select(ci => (TensorPrimitives.Dot(clusterWord.OldVectorNormalized, ci.CentroidOldVectorNormalized), ci))
                    .OrderByDescending(it => it.Item1)
                    .Take(8)
                    .ToArray();
                Temp_Top8ProxClusterInfos[clusterWord.Index] = temp_Top8ProxClusterInfos;                
            }                
        }
        
        Temp_DependentWords = new Word[dependentWords.Length][];  
        foreach (int wordIndex in Enumerable.Range(0, dependentWords.Length))
        {
            Temp_DependentWords[wordIndex] = dependentWords[wordIndex].Distinct().ToArray();
        }

        Temp_InRangeDataIndicesCollection = new DenseMatrix<bool>(wordsCount, wordsCount);
        float delta = 0.05f;
        int count = (int)((1.0f - 0.5f) / delta);
        Temp_RangeDataIndicesCollection = new int[count][];
        float high = 1.0f;
        for (int rangeIndex = count - 1; rangeIndex >= 0; rangeIndex -= 1)
        {
            float low = high - delta;
            var rangeDataIndices = proxWordsOldMatrix.Data.Select((f, i) => (f, i))
                .Where(it => low <= it.f && it.f < high)
                .Select(it => it.i)
                .ToArray();
            foreach (int dataIndex in rangeDataIndices)
            {
                Temp_InRangeDataIndicesCollection.Data[dataIndex] = true;
            }
            Temp_RangeDataIndicesCollection[rangeIndex] = rangeDataIndices;
            high = low;
        }
    } 
    
    /// <summary>
    ///     Calculates all vectors and matrices.
    /// </summary>
    /// <param name="words"></param>
    /// <param name="wordsHashProjectionIndices"></param>
    /// <param name="loggersSet"></param>
    public void Calculate_DiscreteVectorsAndMatrices(List<Word> words,
        int[] wordsHashProjectionIndices, 
        ILoggersSet loggersSet)
    {
        var wordsSubArray = words.ToArray();

        CalculateDiscreteVectorsOnly(wordsSubArray,
            wordsHashProjectionIndices,
            loggersSet);

        var stopwatch = Stopwatch.StartNew();

        ParallelLoopResult parallelLoopResult = Parallel.For(0, wordsSubArray.Length, i1 =>
        {
            int wordIndex1 = wordsSubArray[i1].Index;
            int indexBias = wordIndex1 * words.Count;

            var discreteVector = DiscreteVectors[wordIndex1];
            var discreteVector_PrimaryOnly = DiscreteVectors_PrimaryBitsOnly[wordIndex1];
            var discreteVector_SecondaryOnly = DiscreteVectors_SecondaryBitsOnly[wordIndex1];
            for (var i2 = 0; i2 < wordsSubArray.Length; i2 += 1)
            {
                int wordIndex2 = wordsSubArray[i2].Index;
                int matrixIndex = indexBias + wordIndex2;
                ProxWordsDiscreteMatrix[matrixIndex] = TensorPrimitives.Dot(discreteVector, DiscreteVectors[wordIndex2]);
                ProxWordsDiscreteMatrix_PrimaryBitsOnly[matrixIndex] = TensorPrimitives.Dot(discreteVector_PrimaryOnly, DiscreteVectors_PrimaryBitsOnly[wordIndex2]);
                ProxWordsDiscreteMatrix_SecondaryBitsOnly[matrixIndex] = TensorPrimitives.Dot(discreteVector_SecondaryOnly, DiscreteVectors_SecondaryBitsOnly[wordIndex2]);
            }
        });

        loggersSet.UserFriendlyLogger.LogInformation("Calculate_Full done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    /// <summary>
    ///     Calculates all vectors and partially matrices.
    /// </summary>
    /// <param name="wordsSubArray"></param>
    /// <param name="words"></param>
    /// <param name="wordsHashProjectionIndices"></param>
    /// <param name="loggersSet"></param>
    public void Calculate_DiscreteVectorsAndMatricesPartial(Word[] wordsSubArray,
        List<Word> words,
        int[] wordsHashProjectionIndices,
        ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        CalculateDiscreteVectorsOnly(wordsSubArray,
            wordsHashProjectionIndices,
            loggersSet);

        var temp_InRangeDataIndicesCollection_Data = Temp_InRangeDataIndicesCollection.Data;

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
                int dataIndex = indexBias + wordIndex2;
                if (temp_InRangeDataIndicesCollection_Data[dataIndex])
                    ProxWordsDiscreteMatrix[wordIndex2 * words.Count + wordIndex1] = 
                        ProxWordsDiscreteMatrix[dataIndex] = 
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
    /// <param name="wordsHashProjectionIndices"></param>
    /// <param name="loggersSet"></param>
    public void CalculateDiscreteVectorsOnly(Word[] wordsSubArray,            
        int[] wordsHashProjectionIndices,
        ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        foreach (Word word in wordsSubArray)
        {
            var discreteVector = DiscreteVectors[word.Index];
            var discreteVector_PrimaryOnly = DiscreteVectors_PrimaryBitsOnly[word.Index];
            var discreteVector_SecondaryOnly = DiscreteVectors_SecondaryBitsOnly[word.Index];
            Array.Clear(discreteVector);
            Array.Clear(discreteVector_PrimaryOnly);
            Array.Clear(discreteVector_SecondaryOnly);
            var temp_Top8ProxPrimaryWords = Temp_Top8ProxClusterInfos[word.Index];
            var temp_Top8ProxWords = Temp_Top8ProxWords[word.Index];
            for (int i = 0; i < temp_Top8ProxPrimaryWords!.Length; i += 1)
            {
                int clusterHashProjectionIndex = wordsHashProjectionIndices[temp_Top8ProxPrimaryWords[i].Item2.HashProjectionIndex];
                if (clusterHashProjectionIndex >= 0)
                {
                    discreteVector[clusterHashProjectionIndex] = 1.0f;
                    discreteVector_PrimaryOnly[clusterHashProjectionIndex] = 1.0f;
                }
            }
            for (int i = 0; i < temp_Top8ProxWords.Length; i += 1)
            {
                int wordHashProjectionIndex = wordsHashProjectionIndices[temp_Top8ProxWords[i].Item2.Index];
                if (wordHashProjectionIndex >= 0)
                {
                    discreteVector[wordHashProjectionIndex] = 1.0f;
                    discreteVector_SecondaryOnly[wordHashProjectionIndex] = 1.0f;
                }
            }
        }

        stopwatch.Stop();
        //loggersSet.UserFriendlyLogger.LogInformation("CalculateDiscreteVectors done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.Write(DiscreteVectors.Length);
            if (DiscreteVectors is not null)
                foreach (var vectorNew in DiscreteVectors)
                {
                    writer.WriteArray(vectorNew);
                }
            //writer.WriteArray(algorithmData.ProxWordsDiscreteMatrix);

            writer.Write(DiscreteVectors_PrimaryBitsOnly.Length);
            if (DiscreteVectors_PrimaryBitsOnly is not null)
                foreach (var vectorNew in DiscreteVectors_PrimaryBitsOnly)
                {
                    writer.WriteArray(vectorNew);
                }
            //writer.WriteArray(algorithmData.ProxWordsDiscreteMatrix_PrimaryOnly);

            writer.Write(DiscreteVectors_SecondaryBitsOnly.Length);
            if (DiscreteVectors_SecondaryBitsOnly is not null)
                foreach (var vectorNew in DiscreteVectors_SecondaryBitsOnly)
                {
                    writer.WriteArray(vectorNew);
                }
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        int wordsCount = 0;
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    int discreteVectorsLength = reader.ReadInt32();
                    wordsCount = discreteVectorsLength;
                    if (discreteVectorsLength > 0)
                    {
                        var discreteVectors = new float[discreteVectorsLength][];
                        foreach (int i in Enumerable.Range(0, discreteVectorsLength))
                        {
                            discreteVectors[i] = reader.ReadArray<float>()!;
                        }
                        DiscreteVectors = discreteVectors;
                    }
                    //algorithmData.ProxWordsDiscreteMatrix = reader.ReadArray<float>();

                    discreteVectorsLength = reader.ReadInt32();
                    if (discreteVectorsLength > 0)
                    {
                        var discreteVectors_PrimaryOnly = new float[discreteVectorsLength][];
                        foreach (int i in Enumerable.Range(0, discreteVectorsLength))
                        {
                            discreteVectors_PrimaryOnly[i] = reader.ReadArray<float>()!;
                        }
                        DiscreteVectors_PrimaryBitsOnly = discreteVectors_PrimaryOnly;
                    }
                    //algorithmData.ProxWordsDiscreteMatrix_PrimaryOnly = reader.ReadArray<float>();

                    discreteVectorsLength = reader.ReadInt32();
                    if (discreteVectorsLength > 0)
                    {
                        var discreteVectors_SecondaryOnly = new float[discreteVectorsLength][];
                        foreach (int i in Enumerable.Range(0, discreteVectorsLength))
                        {
                            discreteVectors_SecondaryOnly[i] = reader.ReadArray<float>()!;
                        }
                        DiscreteVectors_SecondaryBitsOnly = discreteVectors_SecondaryOnly;
                    }
                    //algorithmData.ProxWordsDiscreteMatrix_SecondaryOnly = reader.ReadArray<float>();
                    break;
            }
        }

        ProxWordsDiscreteMatrix = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_PrimaryBitsOnly = new float[wordsCount * wordsCount];
        ProxWordsDiscreteMatrix_SecondaryBitsOnly = new float[wordsCount * wordsCount];
    }
}

// Word[] except_WordsSubArray = words.Except(wordsSubArray).ToArray();