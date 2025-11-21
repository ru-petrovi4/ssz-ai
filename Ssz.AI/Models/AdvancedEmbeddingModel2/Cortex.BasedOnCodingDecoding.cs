using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils;
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
    public const int ReorderPhrases_BasedOnCodingDecoding_CortexMemories_BatchSize = 1000000;
    public const int ReorderPhrases_BasedOnCodingDecoding_CortexMemories_MaxCount = Int32.MaxValue;

    #region public functions

    public async Task Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        int cortexMemories_Count = Math.Min(ReorderPhrases_BasedOnCodingDecoding_CortexMemories_MaxCount, CortexMemories.Count);

        int cortexMemories_BatchesCount = cortexMemories_Count / ReorderPhrases_BasedOnCodingDecoding_CortexMemories_BatchSize + 1;

        int inMiniColumn_MaxMemoriesCount = 1000000 / MiniColumns.Data.Length;

        var random_CortexMemories = CortexMemories.Take(cortexMemories_Count).ToArray();

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
                var k_ForNearestMiniColumns = new FastList<(float, float, IMiniColumnActivity)>((int)(Math.PI * Constants.PositiveK.Length * Constants.PositiveK.Length) + 10);

                k_ForNearestMiniColumns.Add((Constants.PositiveK[0], Constants.NegativeK[0], miniColumnActivity));

                for (int i = 1; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
                {
                    var it = miniColumn.Temp_K_ForNearestMiniColumns[i];
                    k_ForNearestMiniColumns.Add((it.Item1, it.Item2, miniColumnActivities[((MiniColumn)it.Item3).MCX, ((MiniColumn)it.Item3).MCY]));
                }

                miniColumnActivity.K_ForNearestMiniColumns = k_ForNearestMiniColumns;                
            }

            cortexMemory.Temp_MiniColumnActivities = miniColumnActivities;

            MiniColumnsActivityHelper.CalculateActivityAndSuperActivity(cortexMemory.DiscreteRandomVector, miniColumnActivities, null, Constants);
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
                    inMiniColumn_MaxMemoriesCount,
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
        int inMiniColumn_MaxMemoriesCount,
        Random random, 
        CancellationToken cancellationToken, 
        Func<Task> refreshAction)
    {
        int changesCount = 0;

        for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var miniColumn = MiniColumns.Data[mci];

            for (int mi = miniColumn.CortexMemories.Count - 1; mi >= Math.Max(0, miniColumn.CortexMemories.Count - inMiniColumn_MaxMemoriesCount); mi -= 1)
            {
                Memory? cortexMemory = miniColumn.CortexMemories[mi];
                if (cortexMemory is null)
                    continue;

                miniColumn.CortexMemories[mi] = null;

                MiniColumn? winnerMiniColumn = null;//GetWinnerMiniColumn_;

                if (winnerMiniColumn is not null)
                {
                    if (!ReferenceEquals(winnerMiniColumn, miniColumn))
                    {
                        winnerMiniColumn.AddCortexMemory(cortexMemory);
                        changesCount += 1;
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

        return changesCount;
    }

    #endregion
}
