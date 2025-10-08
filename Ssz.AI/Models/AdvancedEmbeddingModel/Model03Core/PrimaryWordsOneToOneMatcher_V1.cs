using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors; // Для TensorPrimitives

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

/// <summary>
/// Сопоставление на основе генерации гипотез
/// </summary>
public class PrimaryWordsOneToOneMatcher_V1
{
    public PrimaryWordsOneToOneMatcher_V1(IUserFriendlyLogger userFriendlyLogger, Model03.Parameters parameters)
    {
        _userFriendlyLogger = userFriendlyLogger;
        _parameters = parameters;

        VectorLength = Model01.Constants.DiscreteVectorLength;
        var d = Model01.Constants.DiscreteVectorLength;
        DistanceMatrixA = new MatrixFloat(d, d);
        DistanceMatrixB = new MatrixFloat(d, d);
        HypothesisSupport = new MatrixFloat(d, d);
    }

    public int VectorLength;

    /// <summary>
    ///     Матрицы близости: [позиция, позиция] -> float
    /// </summary>
    public MatrixFloat DistanceMatrixA;

    public Nearest NearestA = null!;

    /// <summary>
    ///     Матрицы близости: [позиция, позиция] -> float
    /// </summary>
    public MatrixFloat DistanceMatrixB;

    public Nearest NearestB = null!;

    /// <summary>
    ///     Таблица накопления весов для гипотез: [позицияA, позицияB] -> float
    /// </summary>
    public MatrixFloat HypothesisSupport;

    /// <summary>
    ///     Заполнить матрицы близости по множеству векторов (0/1 значения)
    /// </summary>
    /// <param name="allExamples"></param>
    /// <param name="distanceMatrix"></param>
    public void BuildDistanceMatrix_V1(MatrixFloat allExamples, MatrixFloat distanceMatrix)
    {   
        var count = allExamples.Dimensions[1];

        // Для каждой позиции i считаем схожесть с позицией j по всем примерам
        for (int i = 0; i < VectorLength; i++)
            for (int j = 0; j < VectorLength; j++)
            {
                float dot = 0, normI = 0, normJ = 0;
                for (int e = 0; e < count; e++)
                {
                    var v = allExamples.GetColumn(e);
                    float vi = v[i];
                    float vj = v[j];
                    dot += vi * vj;
                    normI += vi;
                    normJ += vj;
                }
                // Для бинарных векторов попарная корреляция
                if (normI > 0 && normJ > 0)
                    distanceMatrix[i, j] = dot / MathF.Sqrt(normI * normJ);
                else
                    distanceMatrix[i, j] = 0;
            }
    }

    /// <summary>
    ///     Заполнить матрицы близости по множеству векторов (0/1 значения)
    /// </summary>
    /// <param name="allExamples"></param>
    /// <param name="distanceMatrix"></param>
    public void BuildDistanceMatrix_V2(MatrixFloat allExamples, MatrixFloat distanceMatrix)
    {
        var count = allExamples.Dimensions[1];

        // Для каждой позиции i считаем схожесть с позицией j по всем примерам
        for (int i = 0; i < VectorLength; i++)
            for (int j = 0; j < VectorLength; j++)
            {
                float dot = 0;
                for (int e = 0; e < count; e++)
                {
                    var v = allExamples.GetColumn(e);
                    float vi = v[i];
                    float vj = v[j];
                    dot += vi * vj;                    
                }
                // Для бинарных векторов попарная корреляция
                distanceMatrix[i, j] = dot;                
            }
    }

    /// <summary>
    ///     Найти ближайшие позиции для всех (по строкам)
    /// </summary>
    /// <param name="distanceMatrix"></param>
    /// <returns></returns>
    public Nearest BuildNearest(MatrixFloat distanceMatrix)
    {
        var array = new FastList<int>[VectorLength];
        for (int i = 0; i < VectorLength; i++)
        {
            var list = new List<(int idx, float val)>(VectorLength);
            for (int j = 0; j < VectorLength; j++)
            {
                if (i != j)
                    list.Add((j, distanceMatrix[i, j]));
            }            
            // Берём NearestCount ближайших
            array[i] = new FastList<int>(list.OrderByDescending(it => it.val).Take(_parameters.NearestCount).Select(x => x.idx).ToArray());
        }
        return new Nearest()
        {
            Array = array
        };
    }

    /// <summary>
    /// Подкрепление гипотез на примерах
    /// </summary>
    /// <param name="setA"></param>
    /// <param name="setB"></param>
    public void SupportHypotheses_V1(MatrixFloat setA, MatrixFloat setB, int? count = 0)
    {   
        var nearestA = NearestA.Array;
        var nearestB = NearestB.Array;
        int finalCount = count ?? setA.Dimensions[1];
        for (int i = 0; i < finalCount; i += 1)
        {
            if (i % 100 == 0)
                _userFriendlyLogger.LogInformation($"A i = {i}");

            var vecA = setA.GetColumn(i);
            // Индексы позиций с единицей
            for (int idxA = 0; idxA < VectorLength; idxA += 1)
            {
                if (vecA[idxA] > 0.5f)
                {
                    for (int idxB = 0; idxB < VectorLength; idxB += 1)
                    {
                        // Гипотеза: idxA → idxB
                        //HypothesisSupport[idxA, idxB] += 1.0f;

                        // Подкрепляем также все пары в 16 ближайших
                        foreach (var nearB in nearestB[idxB].Items)
                        {
                            for (int idxA2 = 0; idxA2 < VectorLength; idxA2 += 1)
                            {
                                if (idxA2 != idxA && vecA[idxA2] > 0.5f)
                                {
                                    HypothesisSupport[idxA2, nearB] += 1.0f;
                                }
                            }
                        }
                    }
                }
            }
        }

        //finalCount = count ?? setA.Dimensions[1];
        //for (int i = 0; i < finalCount; i += 1)
        //{
        //    if (i % 100 == 0)
        //        _userFriendlyLogger.LogInformation($"B i = {i}");

        //    var vecB = setB.GetColumn(i);
        //    // Индексы позиций с единицей
        //    for (int idxB = 0; idxB < VectorLength; idxB += 1)
        //    {
        //        if (vecB[idxB] > 0.5f)
        //        {
        //            for (int idxA = 0; idxA < VectorLength; idxA += 1)
        //            {
        //                // Гипотеза: idxA → idxB
        //                //HypothesisSupport[idxA, idxB] += 1.0f;

        //                // Подкрепляем также все пары в 16 ближайших
        //                foreach (var nearA in nearestA[idxA].Items)
        //                {
        //                    for (int idxB2 = 0; idxB2 < VectorLength; idxB2 += 1)
        //                    {
        //                        if (idxB2 != idxB && vecB[idxB2] > 0.5f)
        //                        {
        //                            HypothesisSupport[nearA, idxB2] += 1.0f;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}
    }

    /// <summary>
    /// Подкрепление гипотез на примерах
    /// </summary>
    /// <param name="setA"></param>
    /// <param name="setB"></param>
    public void SupportHypotheses_V2(MatrixFloat setA, MatrixFloat setB, int? count = 0)
    {
        var nearestA = NearestA.Array;
        var nearestB = NearestB.Array;
        int finalCount = count ?? setA.Dimensions[1];
        for (int i = 0; i < finalCount; i += 1)
        {
            if (i % 100 == 0)
                _userFriendlyLogger.LogInformation($"A i = {i}");

            var vecA = setA.GetColumn(i);

            for (int idxB = 0; idxB < VectorLength; idxB += 1)
            {
                for (int idxA2 = 0; idxA2 < VectorLength; idxA2 += 1)
                {
                    if (vecA[idxA2] > 0.5f)
                    {
                        HypothesisSupport[idxA2, idxB] += 1.0f;
                    }
                }

                // Подкрепляем также все пары в 16 ближайших
                foreach (var nearB in nearestB[idxB].Items)
                {
                    for (int idxA2 = 0; idxA2 < VectorLength; idxA2 += 1)
                    {
                        if (vecA[idxA2] > 0.5f)
                        {
                            HypothesisSupport[idxA2, nearB] += 1.0f;
                        }
                    }
                }
            }
        }

        //finalCount = count ?? setA.Dimensions[1];
        //for (int i = 0; i < finalCount; i += 1)
        //{
        //    if (i % 100 == 0)
        //        _userFriendlyLogger.LogInformation($"B i = {i}");

        //    var vecB = setB.GetColumn(i);
        //    // Индексы позиций с единицей
        //    for (int idxB = 0; idxB < VectorLength; idxB += 1)
        //    {
        //        if (vecB[idxB] > 0.5f)
        //        {
        //            for (int idxA = 0; idxA < VectorLength; idxA += 1)
        //            {
        //                // Гипотеза: idxA → idxB
        //                //HypothesisSupport[idxA, idxB] += 1.0f;

        //                // Подкрепляем также все пары в 16 ближайших
        //                foreach (var nearA in nearestA[idxA].Items)
        //                {
        //                    for (int idxB2 = 0; idxB2 < VectorLength; idxB2 += 1)
        //                    {
        //                        if (idxB2 != idxB && vecB[idxB2] > 0.5f)
        //                        {
        //                            HypothesisSupport[nearA, idxB2] += 1.0f;
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}
    }

    // Получить итоговое соответствие
    public int[] GetFinalMappingForcedExclusive()
    {
        var result = new int[VectorLength];
        var usedB = new HashSet<int>();

        for (int i = 0; i < VectorLength; i++)
        {
            // Ищем позицию B с максимальным весом среди неиспользованных
            float max = float.MinValue;
            int selected = -1;
            for (int j = 0; j < VectorLength; j++)
            {
                if (!usedB.Contains(j) && HypothesisSupport[i, j] > max)
                {
                    max = HypothesisSupport[i, j];
                    selected = j;
                }
            }
            if (selected != -1)
            {
                result[i] = selected;
                usedB.Add(selected);
            }
        }
        return result;
    }

    public int[] GetFinalMapping()
    {
        var result = new int[VectorLength];        

        for (int i = 0; i < VectorLength; i++)
        {
            // Ищем позицию B с максимальным весом среди неиспользованных
            float max = float.MinValue;
            int selected = -1;
            for (int j = 0; j < VectorLength; j++)
            {
                if (HypothesisSupport[i, j] > max)
                {
                    max = HypothesisSupport[i, j];
                    selected = j;
                }
            }
            if (selected != -1)
            {
                result[i] = selected;                
            }
        }
        return result;
    }

    #region private fields

    private IUserFriendlyLogger _userFriendlyLogger;
    private Model03.Parameters _parameters;

    #endregion

    public class Nearest : IOwnedDataSerializable
    {
        public FastList<int>[] Array = null!;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArrayOfOwnedDataSerializable(Array, context);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        Array = reader.ReadArrayOfOwnedDataSerializable(() => new FastList<int>(0), context);
                        break;
                }
            }
        }
    }
}

