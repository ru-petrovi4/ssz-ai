using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.CortexVisualisationModel;

public interface IMiniColumnsActivityConstants
{
    int CotrexWidth_MiniColumns { get; }

    int CotrexHeight_MiniColumns { get; }

    /// <summary>
    ///     Олпределенный зараниие радиус гиперколонки в миниколонках.
    /// </summary>
    int HypercolumnDefinedRadius_MiniColumns { get; }

    /// <summary>
    ///     Уровень подобия для нулевой активности
    /// </summary>
    float K0 { get; set; }

    /// <summary>
    ///     Уровень подобия с пустой миниколонкой
    /// </summary>
    float K2 { get; set; }

    /// <summary>
    ///     Порог суперактивности
    /// </summary>
    float K4 { get; set; }

    float[] PositiveK { get; set; }

    float[] NegativeK { get; set; }

    /// <summary>
    ///     Включен ли порог на суперактивность при накоплении воспоминаний
    /// </summary>
    public bool SuperactivityThreshold { get; set; }
}

public class ActivitiyMaxInfo
{
    public Cortex.MiniColumn? SelectedSuperActivityMax_MiniColumn;

    public float MaxActivity = float.MinValue;
    public readonly List<Cortex.MiniColumn> ActivityMax_MiniColumns = new();

    public float MaxSuperActivity = float.MinValue;
    public readonly List<Cortex.MiniColumn> SuperActivityMax_MiniColumns = new();

    public Cortex.MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
    {
        if (SuperActivityMax_MiniColumns.Count == 0)
            SelectedSuperActivityMax_MiniColumn = null;
        else if (SuperActivityMax_MiniColumns.Count == 1)
            SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[0];
        else
            SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[random.Next(SuperActivityMax_MiniColumns.Count)];
        return SelectedSuperActivityMax_MiniColumn;
    }
}

public static class MiniColumnsActivityHelper
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
        IMiniColumnsActivityConstants constants)
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

    /// <summary>
    ///     
    /// </summary>
    /// <param name="miniColumnActivity"></param>
    /// <param name="constants"></param>
    /// <returns></returns>
    public static float GetSuperActivity(Cortex.MiniColumn miniColumn, IMiniColumnsActivityConstants constants)
    {
        float superActivity;
        
        if (miniColumn.Temp_Activity.CortexMemoriesCount > 0)
            superActivity = constants.PositiveK[0] * miniColumn.Temp_Activity.PositiveActivity +
                constants.NegativeK[0] * miniColumn.Temp_Activity.NegativeActivity;
        else
            superActivity = constants.PositiveK[0] * (constants.K2 - constants.K0); // Best proximity

        for (int i = 0; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
        {
            var it = miniColumn.Temp_K_ForNearestMiniColumns[i];
            var nearestMiniColumn = it.MiniColumn;

            if (nearestMiniColumn.Temp_Activity.CortexMemoriesCount > 0)
                superActivity += it.PositiveK * nearestMiniColumn.Temp_Activity.PositiveActivity +
                    it.NegativeK * nearestMiniColumn.Temp_Activity.NegativeActivity;           
        }

        return superActivity;
    }
}        