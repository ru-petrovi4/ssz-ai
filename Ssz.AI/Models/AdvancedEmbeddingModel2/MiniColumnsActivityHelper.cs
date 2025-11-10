using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public static class MiniColumnsActivityHelper
{
    /// <summary>
    ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина), количество воспоминаний
    ///     Всегда не NaN
    /// </summary>
    /// <param name="discreteRandomVector"></param>
    /// <returns></returns>
    public static (float PositiveActivity, float NegativeActivity, int MemoriesCount) GetActivity(Cortex.MiniColumn miniColumn, float[] discreteRandomVector, Model01.ModelConstants constants)
    {
        float positiveActivity = 0.0f;
        int positiveMemoriesCount = 0;
        float negativeActivity = 0.0f;
        int negativeMemoriesCount = 0;

        foreach (var mi in Enumerable.Range(0, miniColumn.CortexMemories.Count))
        {
            var cortexMemory = miniColumn.CortexMemories[mi];
            if (cortexMemory is null)
                continue;

            float memoryCosineSimilarity = TensorPrimitives.CosineSimilarity(discreteRandomVector, cortexMemory.DiscreteRandomVector);
            //if (memoryCosineSimilarity > constants.K1)
            {
                float activity = memoryCosineSimilarity - constants.K0;
                if (activity >= 0)
                {
                    positiveActivity += activity;
                    positiveMemoriesCount += 1;
                }
                else
                {
                    negativeActivity += activity;
                    negativeMemoriesCount += 1;
                }
            }
        }

        if (positiveMemoriesCount > 0)
            positiveActivity /= positiveMemoriesCount;

        if (negativeMemoriesCount > 0)
            negativeActivity /= negativeMemoriesCount;

        return (positiveActivity, negativeActivity, positiveMemoriesCount + negativeMemoriesCount);
    }

    public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Model01.ModelConstants constants)
    {            
        float superActivity;

        if (miniColumn.Temp_Activity.MemoriesCount > 0)
            superActivity = miniColumn.K0.Item1 * miniColumn.Temp_Activity.PositiveActivity + miniColumn.K0.Item2 * miniColumn.Temp_Activity.NegativeActivity;
        else
            superActivity = miniColumn.K0.Item1 * (constants.K2 - constants.K0); // Best proximity

        foreach (var it in miniColumn.K_ForNearestMiniColumns)
        {
            var nearestMiniColumn = it.Item3;                

            if (nearestMiniColumn.Temp_Activity.MemoriesCount > 0)
                superActivity += it.Item1 * nearestMiniColumn.Temp_Activity.PositiveActivity +
                    it.Item2 * nearestMiniColumn.Temp_Activity.NegativeActivity;
            //else
            //    superActivity += it.Item1 * (constants.K2 - constants.K0); // Best proximity
        }

        return superActivity;
    }
}