using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public partial class Cortex : ISerializableModelObject
{
    #region public functions

    /// <summary>
    ///     Returns true when finished.
    /// </summary>
    /// <param name="inputCorpusData"></param>
    /// <param name="cortexMemoriesCount"></param>
    /// <param name="random"></param>
    /// <returns></returns>
    public bool Calculate_PutPhrases_BasedOnSuperActivity(InputCorpusData inputCorpusData, int cortexMemoriesCount, Random random)
    {
        try
        {
            for (int i = 0; i < cortexMemoriesCount; i += 1)
            {
                if (inputCorpusData.CurrentCortexMemoryIndex >= inputCorpusData.CortexMemories.Count - 1)
                    return true;

                inputCorpusData.CurrentCortexMemoryIndex += 1;

                var cortexMemory = inputCorpusData.CortexMemories[inputCorpusData.CurrentCortexMemoryIndex];

                Temp_InputCurrentDesc = GetDesc(cortexMemory);
                CalculateActivityAndSuperActivity(cortexMemory.DiscreteRandomVector, Temp_ActivitiyMaxInfo);

                MiniColumn? winnerMiniColumn;
                // Сохраняем воспоминание в миниколонке-победителе.
                //if (randomInitialization)
                //{
                //    var winnerIndex = random.Next(MiniColumns.Length);
                //    winnerMiniColumn = MiniColumns[winnerIndex];
                //}
                //else
                {
                    winnerMiniColumn = Temp_ActivitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
                }
                if (winnerMiniColumn is not null)
                {
                    winnerMiniColumn.AddCortexMemory(cortexMemory);
                }
            }
            return false;
        }
        finally
        {
            Logger.LogInformation($"CalculateCortexMemories() {inputCorpusData.CurrentCortexMemoryIndex}/{inputCorpusData.CortexMemories.Count} finished.");
        }
    }

    public async Task Calculate_ReorderCortexMemories_BasedOnSuperActivityAsync(int epochCount, Random random, Func<Task>? epochRefreshAction = null)
    {
        ActivitiyMaxInfo activitiyMaxInfo = new();
        int min_EpochChangesCount = Int32.MaxValue;

        int maxMemoriesCount = 10000 / MiniColumns.Data.Length;

        Stopwatch sw = new();
        for (int epochIndex = 0; epochIndex < epochCount; epochIndex += 1)
        {
            sw.Restart();

            int epochChangesCount = 0;
            for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
            {
                MiniColumn miniColumn = MiniColumns.Data[mci];

                for (int mi = miniColumn.CortexMemories.Count - 1; mi >= Math.Max(0, miniColumn.CortexMemories.Count - maxMemoriesCount); mi -= 1)
                {
                    Memory? cortexMemory = miniColumn.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;

                    miniColumn.CortexMemories[mi] = null;

                    Temp_InputCurrentDesc = GetDesc(cortexMemory);
                    CalculateActivityAndSuperActivity(cortexMemory.DiscreteRandomVector, activitiyMaxInfo);

                    // Сохраняем воспоминание в миниколонке-победителе.
                    MiniColumn? winnerMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
                    if (winnerMiniColumn is not null)
                    {
                        if (!ReferenceEquals(winnerMiniColumn, miniColumn))
                        {
                            winnerMiniColumn.AddCortexMemory(cortexMemory);
                            epochChangesCount += 1;
                        }
                        else
                        {
                            miniColumn.CortexMemories[mi] = cortexMemory;
                        }
                    }
                }
            }

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

            sw.Stop();

            Logger.LogInformation($"ReorderMemories() epoch {epochIndex + 1}/{epochCount} finished. ChangedCount: {epochChangesCount}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}");

            if (epochChangesCount < min_EpochChangesCount)
            {
                min_EpochChangesCount = epochChangesCount;
            }

            if (epochChangesCount < 10)
            {
                break;
            }
            else
            {
                if (epochRefreshAction is not null)
                    await epochRefreshAction();
            }
        }
    }

    #endregion

    #region private functions

    #endregion
}
