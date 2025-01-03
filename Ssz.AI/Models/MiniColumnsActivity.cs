using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models
{
    public static class MiniColumnsActivity
    {
        /// <summary>
        ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина)
        /// </summary>
        /// <param name="hash"></param>
        /// <returns></returns>
        public static (float, float) GetActivity(Cortex_WithSubarea.MiniColumn miniColumn, float[] hash)
        {
            if (TensorPrimitives.Sum(hash) < miniColumn.Constants.MinBitsInHashForMemory)
                //return (0.0f, 0.0f);
                return (float.NaN, float.NaN);

            float positiveActivity = 0.0f;
            int positive_MemoryCount = 0;

            float negativeActivity = 0.0f;
            int negative_MemoryCount = 0;

            foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
            {
                var memory = miniColumn.Memories[mi];
                if (memory.IsDeleted)
                    continue;
                float a = TensorPrimitives.CosineSimilarity(hash, memory.Hash) - 0.66f;
                if (a > 0.0f)
                {
                    positiveActivity += a;
                    positive_MemoryCount += 1;
                }
                else if (a > -0.33)
                {
                    negativeActivity += a;
                    negative_MemoryCount += 1;
                }

                if (float.IsNaN(positiveActivity) ||
                        float.IsNaN(negativeActivity))
                    throw new Exception();
            }

            //if (positive_MemoryCount > 0)
            //    positiveActivity = positiveActivity / positive_MemoryCount;

            //if (negative_MemoryCount > 0)
            //    negativeActivity = negativeActivity / negative_MemoryCount;

            return (positiveActivity, negativeActivity);
        }

        public static float GetSuperActivity(Cortex_WithSubarea.MiniColumn miniColumn)
        {
            //float superActivity = Temp_Activity.Item1 + Temp_Activity.Item2;
            float positiveActivitySum = miniColumn.Temp_Activity.Item1;
            float positiveActivitySum_TotalK = 1.0f;

            float negativeActivitySum = miniColumn.Temp_Activity.Item2;
            float negativeActivitySum_TotalK = 1.0f;

            foreach (var r in Enumerable.Range(0, miniColumn.NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = miniColumn.NearestMiniColumnInfos[r];

                int nearestMiniColumnsForRCount = 0;
                float positiveActivitySumForR = 0.0f;
                float negativeActivitySumForR = 0.0f;
                foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Item2.Count))
                {
                    var nearestMiniColumnInfo = nearestMiniColumnInfosForR.Item2[mci];
                    if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item1))
                    {
                        positiveActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item1;
                        negativeActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item2;
                        nearestMiniColumnsForRCount += 1;
                    }
                }

                if (nearestMiniColumnsForRCount == 0)
                    continue;

                //superActivity += (nearestMiniColumnInfosForR.Item1.Item1 * activitySumForR) + (nearestMiniColumnInfosForR.Item1.Item2 * antiActivitySumForR);

                positiveActivitySum += nearestMiniColumnInfosForR.Item1.Item1 * positiveActivitySumForR / nearestMiniColumnsForRCount;
                positiveActivitySum_TotalK += nearestMiniColumnInfosForR.Item1.Item1;

                negativeActivitySum += nearestMiniColumnInfosForR.Item1.Item2 * negativeActivitySumForR / nearestMiniColumnsForRCount;
                negativeActivitySum_TotalK += nearestMiniColumnInfosForR.Item1.Item2;
            }

            return (positiveActivitySum / positiveActivitySum_TotalK) + (negativeActivitySum / negativeActivitySum_TotalK);
            //return superActivity;
        }

        public class ActivitiyMaxInfo
        {
            public float MaxActivity = float.MinValue;
            public readonly List<Cortex_WithSubarea.MiniColumn> ActivityMax_MiniColumns = new();
            public Cortex_WithSubarea.MiniColumn? GetActivityMax_MiniColumn(Random random)
            {
                if (ActivityMax_MiniColumns.Count == 0)
                    return null;
                if (ActivityMax_MiniColumns.Count == 1)
                    return ActivityMax_MiniColumns[0];
                return ActivityMax_MiniColumns[random.Next(ActivityMax_MiniColumns.Count)];
            }

            public float MaxSuperActivity = float.MinValue;
            public readonly List<Cortex_WithSubarea.MiniColumn> SuperActivityMax_MiniColumns = new();
            public Cortex_WithSubarea.MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
            {
                if (SuperActivityMax_MiniColumns.Count == 0)
                    return null;
                if (SuperActivityMax_MiniColumns.Count == 1)
                    return SuperActivityMax_MiniColumns[0];
                return SuperActivityMax_MiniColumns[random.Next(SuperActivityMax_MiniColumns.Count)];
            }
        }
    }
}
