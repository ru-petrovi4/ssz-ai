using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public partial class Cortex : ISerializableModelObject
{
    public const int ReorderPhrases_BasedOnCodingDecoding_CortexMemories_BatchSize = 1000000;
    public const int ReorderPhrases_BasedOnCodingDecoding_CortexMemories_MaxCount = Int32.MaxValue;

    #region public functions

    public async Task Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        int cortexMemories_Count = Math.Min(ReorderPhrases_BasedOnCodingDecoding_CortexMemories_MaxCount, CortexMemories.Count);

        int cortexMemories_BatchesCount = cortexMemories_Count / ReorderPhrases_BasedOnCodingDecoding_CortexMemories_BatchSize + 1;

        int inMiniColumn_TopMemoriesCount = 1000000 / MiniColumns.Data.Length;

        var random_CortexMemories = CortexMemories.Take(cortexMemories_Count).ToArray();

        int maxR = Constants.PositiveK.Length - 1;

        for (int cmi = 0; cmi < random_CortexMemories.Length; cmi += 1)
        {
            var cortexMemory = random_CortexMemories[cmi];

            var miniColumnActivities = new DenseMatrix<MiniColumnActivity>(MiniColumns.Dimensions);

            for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
            {
                var miniColumn = MiniColumns.Data[mci];
                MiniColumnActivity miniColumnActivity = new MiniColumnActivity(miniColumn);                
                miniColumnActivities[miniColumn.MCX, miniColumn.MCY] = miniColumnActivity;
            }

            for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)                
            {
                var miniColumn = MiniColumns.Data[mci];
                MiniColumnActivity miniColumnActivity = miniColumnActivities[miniColumn.MCX, miniColumn.MCY];
                var k_ForNearestMiniColumns = new FastList<(float, float, IMiniColumnActivity)>((int)(Math.PI * maxR * maxR) + 1);

                k_ForNearestMiniColumns.Add((Constants.PositiveK[0], Constants.NegativeK[0], miniColumnActivity));

                for (int i = 1; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
                {
                    var it = miniColumn.Temp_K_ForNearestMiniColumns[i];
                    k_ForNearestMiniColumns.Add((it.Item1, it.Item2, miniColumnActivities[((MiniColumn)it.Item3).MCX, ((MiniColumn)it.Item3).MCY]));
                }

                miniColumnActivity.K_ForNearestMiniColumns = k_ForNearestMiniColumns;                
            }

            cortexMemory.Temp_MiniColumnActivities = miniColumnActivities;
            cortexMemory.Temp_MiniColumnActivities_PriorityQueue = new PriorityQueue<MiniColumnActivity, float>(Constants.DiscreteOptimizedVector_PrimaryBitsCount);
            cortexMemory.Temp_Int_Single_PriorityQueue = new PriorityQueue<int, float>(60);
            //cortexMemory.Temp_DiscreteOptimizedVector = new float[Constants.DiscreteVectorLength];
            cortexMemory.Temp_DiscreteRandomVector_Restored = new float[Constants.DiscreteVectorLength];            
        }

        Stopwatch sw = new();
        for (int epoch = 0; ; epoch += 1)
        {
            sw.Restart();
            
            random.Shuffle(random_CortexMemories);

            int epochChangesCount = 0;

            for (int cortexMemories_BatchN = 0; cortexMemories_BatchN < cortexMemories_BatchesCount; cortexMemories_BatchN += 1)
            {
                int cortexMemories_IndexStart = cortexMemories_BatchN * ReorderPhrases_BasedOnCodingDecoding_CortexMemories_BatchSize;
                if (cortexMemories_IndexStart >= cortexMemories_Count)
                    break;
                var batch_CortexMemories = random_CortexMemories.AsMemory(cortexMemories_IndexStart, Math.Min(ReorderPhrases_BasedOnCodingDecoding_CortexMemories_BatchSize, random_CortexMemories.Length - cortexMemories_IndexStart));

                epochChangesCount += await Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(
                    batch_CortexMemories,
                    inMiniColumn_TopMemoriesCount,
                    random, 
                    cancellationToken, 
                    refreshAction);
            }            

            sw.Stop();

            Logger.LogInformation($"ReorderPhrases() epoch {epoch + 1}/Max finished. ChangedCount: {epochChangesCount}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}");
            
            if (epochChangesCount < 10)
                break;
        }        
    }

    #endregion

    #region private functions

    private async Task<int> Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(
        Memory<Cortex.Memory> batch_CortexMemories,
        int inMiniColumn_TopMemoriesCount,
        Random random, 
        CancellationToken cancellationToken, 
        Func<Task> refreshAction)
    {
        int changesCount = 0;

        var batch_CortexMemories_Span = batch_CortexMemories.Span;
        for (int cmi = 0; cmi < batch_CortexMemories_Span.Length; cmi += 1)
        {
            var batch_CortexMemory = batch_CortexMemories_Span[cmi];
            MiniColumnsActivityHelper.CalculateActivityAndSuperActivity(
                batch_CortexMemory.DiscreteRandomVector, 
                batch_CortexMemory.Temp_MiniColumnActivities, 
                null, 
                Constants);
        }
        float current_CodingDecodingSimilarity = GetCodingDecodingSimilarity(
                        batch_CortexMemories
                        );

        var sw = new Stopwatch();
        for (int miniColumns_Epoch = 0; ; miniColumns_Epoch += 1) // TEMPCODE
        {
            sw.Restart();

            for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var miniColumn = MiniColumns.Data[mci];

                // TEMPCODE
                //for (int mi = miniColumn.CortexMemories.Count - 1; mi >= Math.Max(0, miniColumn.CortexMemories.Count - inMiniColumn_TopMemoriesCount); mi -= 1)
                // !!! Remove break at the loop end.
                for (; ; )
                {
                    int mi = random.Next(miniColumn.CortexMemories.Count);
                    
                    Memory? cortexMemory = miniColumn.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;

                    miniColumn.CortexMemories[mi] = null;

                    float maxCodingDecodingSimilarity = current_CodingDecodingSimilarity;
                    MiniColumn max_MiniColumn = miniColumn;

                    MiniColumn? prevIteratorMiniColumn = miniColumn;
                    MiniColumn? iteratorMiniColumn = null;
                    for (int i = 1; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
                    {
                        iteratorMiniColumn = (MiniColumn)miniColumn.Temp_K_ForNearestMiniColumns[i].Item3;
                        iteratorMiniColumn.CortexMemories.Add(cortexMemory);

                        var batch_CortexMemories_Span2 = batch_CortexMemories.Span;
                        for (int cmi = 0; cmi < batch_CortexMemories_Span2.Length; cmi += 1)
                        {
                            var batch_CortexMemory = batch_CortexMemories_Span2[cmi];
                            MiniColumnsActivityHelper.ReCalculateActivityAndSuperActivity(
                                batch_CortexMemory.DiscreteRandomVector,
                                batch_CortexMemory.Temp_MiniColumnActivities,
                                Constants,
                                prevIteratorMiniColumn,
                                iteratorMiniColumn);
                            //MiniColumnsActivityHelper.CalculateActivityAndSuperActivity(
                            //    batch_CortexMemory.DiscreteRandomVector,
                            //    batch_CortexMemory.Temp_MiniColumnActivities,
                            //    null,
                            //    Constants);
                        }
                        current_CodingDecodingSimilarity = GetCodingDecodingSimilarity(
                            batch_CortexMemories
                            );

                        if (current_CodingDecodingSimilarity > maxCodingDecodingSimilarity)
                        {
                            maxCodingDecodingSimilarity = current_CodingDecodingSimilarity;
                            max_MiniColumn = iteratorMiniColumn;
                        }

                        iteratorMiniColumn.CortexMemories.RemoveAt(iteratorMiniColumn.CortexMemories.Count - 1);
                        prevIteratorMiniColumn = iteratorMiniColumn;
                    }

                    if (!ReferenceEquals(max_MiniColumn, miniColumn))
                    {
                        max_MiniColumn.AddCortexMemory(cortexMemory);
                        changesCount += 1;
                    }
                    else
                    {
                        miniColumn.CortexMemories[mi] = cortexMemory;
                    }

                    if (!ReferenceEquals(max_MiniColumn, prevIteratorMiniColumn))
                    {
                        var batch_CortexMemories_Span3 = batch_CortexMemories.Span;
                        for (int cmi = 0; cmi < batch_CortexMemories_Span3.Length; cmi += 1)
                        {
                            var batch_CortexMemory = batch_CortexMemories_Span3[cmi];
                            MiniColumnsActivityHelper.ReCalculateActivityAndSuperActivity(
                                batch_CortexMemory.DiscreteRandomVector,
                                batch_CortexMemory.Temp_MiniColumnActivities,
                                Constants,
                                prevIteratorMiniColumn,
                                max_MiniColumn);
                            //MiniColumnsActivityHelper.CalculateActivityAndSuperActivity(
                            //    batch_CortexMemory.DiscreteRandomVector,
                            //    batch_CortexMemory.Temp_MiniColumnActivities,
                            //    null,
                            //    Constants);
                        }
#if DEBUG
                        current_CodingDecodingSimilarity = GetCodingDecodingSimilarity(
                            batch_CortexMemories
                            );
                        if (current_CodingDecodingSimilarity != maxCodingDecodingSimilarity)
                        {
                            throw new InvalidOperationException();
                        }
#else
                        current_CodingDecodingSimilarity = maxCodingDecodingSimilarity;
#endif
                    }

                    break;
                }
            }

            if ((miniColumns_Epoch + 1) % 10 == 0)
            {
                for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
                {
                    MiniColumn miniColumn = MiniColumns.Data[mci];
                    miniColumn.Temp_CortexMemories.Clear();

                    for (int mi = 0; mi < miniColumn.CortexMemories.Count; mi += 1)
                    {
                        Memory? memory = miniColumn.CortexMemories[mi];
                        if (memory is null)
                            continue;

                        miniColumn.Temp_CortexMemories.Add(memory);
                    }

                    miniColumn.CortexMemories.Swap(miniColumn.Temp_CortexMemories);
                    miniColumn.Temp_CortexMemories.Clear();
                }
            }

            sw.Stop();

            Logger.LogInformation($"ReorderPhrases() miniColumns_Epoch {miniColumns_Epoch + 1}/Max finished. current_CodingDecodingSimilarity: {current_CodingDecodingSimilarity}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}");
            await refreshAction();
        }        

        return changesCount;
    }

    //private MiniColumn? GetWinnerMiniColumn_BasedOnCodingDecoding(
    //    Memory<Memory> batch_CortexMemories, 
    //    MiniColumn current_MiniColumn, 
    //    Memory cortexMemory)
    //{
    //    return max_MiniColumn;
    //}

    private float GetCodingDecodingSimilarity(Memory<Memory> batch_CortexMemories)
    {
        int codingDecodingSimilarity_Total = 0;
        Parallel.For(
            0, // начальный индекс (включительно)
            batch_CortexMemories.Length, // конечный индекс (не включительно)
            () => 0, // инициализация локального аккумулятора для потока
            (i, loopState, localSum) => // Основной делегат, исполняемый в параллельном потоке
            {
                var batch_CortexMemory = batch_CortexMemories.Span[i];

                localSum += (int)(GetCodingDecodingSimilarity(batch_CortexMemory) * 10000);

                return localSum;
            },
            localSum => Interlocked.Add(ref codingDecodingSimilarity_Total, localSum)); // объединяем суммы из потоков);
        return ((float)codingDecodingSimilarity_Total / 10000) / batch_CortexMemories.Length;
    }

    private float GetCodingDecodingSimilarity(Memory batch_CortexMemory)
    {
        var k = Constants.DiscreteOptimizedVector_PrimaryBitsCount;
        var pq = batch_CortexMemory.Temp_MiniColumnActivities_PriorityQueue;
        int i = 0;
        for (; i < k; i += 1)
        {
            MiniColumnActivity miniColumnActivity = batch_CortexMemory.Temp_MiniColumnActivities.Data[i];
            float a = miniColumnActivity.SuperActivity; //miniColumnActivity.Activity.PositiveActivity + miniColumnActivity.Activity.NegativeActivity;
            pq.Enqueue(miniColumnActivity, a);
        }
        for (; i < batch_CortexMemory.Temp_MiniColumnActivities.Data.Length; i += 1)
        {
            MiniColumnActivity miniColumnActivity = batch_CortexMemory.Temp_MiniColumnActivities.Data[i];
            float a = miniColumnActivity.SuperActivity; //miniColumnActivity.Activity.PositiveActivity + miniColumnActivity.Activity.NegativeActivity;
            pq.TryPeek(out var minMiniColumnActivity, out var minA);
            if (a > minA)
            {
                pq.Dequeue();
                pq.Enqueue(miniColumnActivity, a);
            }
        }
        var discreteRandomVector_Restored = batch_CortexMemory.Temp_DiscreteRandomVector_Restored;
        Array.Clear(discreteRandomVector_Restored);
        foreach (var item in pq.UnorderedItems)
        {
            //discreteOptimizedVector[item.Element.MiniColumn.DiscreteOptimizedVector_ProjectionIndex] = ;
            var cortexMemories = item.Element.MiniColumn.CortexMemories;
            for (int mi = 0; mi < cortexMemories.Count; mi += 1)
            {
                var cortexMemory = cortexMemories[mi];
                if (cortexMemory is not null)
                    TensorPrimitives.Add(discreteRandomVector_Restored, cortexMemory.DiscreteVector, discreteRandomVector_Restored);
            }            
        }
        pq.Clear();
        int onesCount = (int)TensorPrimitives.Sum(batch_CortexMemory.DiscreteRandomVector);
        MathHelper.SelectTopKMaxAndSetToOne(discreteRandomVector_Restored, onesCount, batch_CortexMemory.Temp_Int_Single_PriorityQueue);
        return TensorPrimitives.CosineSimilarity(batch_CortexMemory.DiscreteRandomVector, discreteRandomVector_Restored);
    }

    #endregion
}
