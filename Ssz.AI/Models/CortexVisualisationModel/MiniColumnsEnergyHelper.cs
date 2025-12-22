using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.CortexVisualisationModel;

public class StateInfo
{
    public Cortex.MiniColumn? SelectedTotalEnergyMin_MiniColumn;

    public float MaxActivity = float.MinValue;
    public readonly List<Cortex.MiniColumn> ActivityMax_MiniColumns = new();

    public float MinTotalEnergy = float.MaxValue;
    public readonly List<Cortex.MiniColumn> TotalEnergyMin_MiniColumns = new();

    public Cortex.MiniColumn? GetTotalEnergyMin_MiniColumn(Random random)
    {
        if (TotalEnergyMin_MiniColumns.Count == 0)
            SelectedTotalEnergyMin_MiniColumn = null;
        else if (TotalEnergyMin_MiniColumns.Count == 1)
            SelectedTotalEnergyMin_MiniColumn = TotalEnergyMin_MiniColumns[0];
        else
            SelectedTotalEnergyMin_MiniColumn = TotalEnergyMin_MiniColumns[random.Next(TotalEnergyMin_MiniColumns.Count)];
        return SelectedTotalEnergyMin_MiniColumn;
    }
}

public static class MiniColumnsEnergyHelper
{
    /// <summary>
    ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина), количество воспоминаний
    ///     Всегда не NaN        
    /// </summary>
    /// <param name="discreteVector"></param>
    /// <returns></returns>
    public static (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) GetActivity(
        Cortex.MiniColumn miniColumn, 
        Cortex.Memory cortexMemory,
        Func<Cortex.Memory, Cortex.Memory, float> getSimilarity,
        ICortexConstants constants)
    {
        float positiveActivity = 0.0f;
        int positiveCortexMemoriesCount = 0;
        float negativeActivity = 0.0f;
        int negativeCortexMemoriesCount = 0;

        for (int mi = 0; mi < miniColumn.CortexMemories.Count; mi += 1)
        {
            var miniColumn_CortexMemory = miniColumn.CortexMemories[mi];
            if (miniColumn_CortexMemory is null)
                continue;

            float similarity = getSimilarity(cortexMemory, miniColumn_CortexMemory);
            if (!Single.IsNaN(similarity))
            {
                float activity = similarity - constants.K0;
                if (activity >= 0)
                {
                    positiveActivity += activity;
                    positiveCortexMemoriesCount += 1;
                }
                else
                {
                    negativeActivity += activity;
                    negativeCortexMemoriesCount += 1;
                }
            }
        }

        if (positiveCortexMemoriesCount > 0)
            positiveActivity /= positiveCortexMemoriesCount;

        if (negativeCortexMemoriesCount > 0)
            negativeActivity /= negativeCortexMemoriesCount;

        return (positiveActivity, negativeActivity, positiveCortexMemoriesCount + negativeCortexMemoriesCount);
    }
    
    public static float GetTotalEnergy(Cortex.MiniColumn miniColumn, ICortexConstants constants)
    {
        float totalEnergy = -constants.PositiveK[0] * miniColumn.Temp_Activity.PositiveActivity -
                constants.NegativeK[0] * miniColumn.Temp_Activity.NegativeActivity -
                (miniColumn.Temp_Activity.CortexMemoriesCount == 0 ? (constants.K2 - constants.K0) : 0.0f); // Штраф за воспоминания        

        for (int i = 0; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
        {
            var it = miniColumn.Temp_K_ForNearestMiniColumns[i];
            var nearestMiniColumn = it.MiniColumn;

            if (nearestMiniColumn.Temp_Activity.CortexMemoriesCount > 0)
                totalEnergy += -it.PositiveK * nearestMiniColumn.Temp_Activity.PositiveActivity -
                    it.NegativeK * nearestMiniColumn.Temp_Activity.NegativeActivity;
        }

        return totalEnergy;
    }
}        