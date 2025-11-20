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

    public async Task Calculate_ReorderPhrases_BasedOnCodingDecodingAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        int maxMemoriesCount = 1000000 / MiniColumns.Data.Length;

        Stopwatch sw = new();
        for (int epochIndex = 0; ; epochIndex += 1)
        {
            sw.Restart();

            int epochChangesCount = 0;

            for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var miniColumn = MiniColumns.Data[mci];

                for (int mi = miniColumn.CortexMemories.Count - 1; mi >= Math.Max(0, miniColumn.CortexMemories.Count - maxMemoriesCount); mi -= 1)
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

            Logger.LogInformation($"ReorderMemories() epoch {epochIndex + 1}/Max finished. ChangedCount: {epochChangesCount}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}");
            
            if (epochChangesCount < 10)
                break;
        }        
    }

    #endregion

    #region private functions

    #endregion
}
