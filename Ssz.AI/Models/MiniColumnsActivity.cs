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
        public static (float, float) GetActivity(Cortex.MiniColumn miniColumn, float[] hash, Cortex cortex)
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
                float a = TensorPrimitives.CosineSimilarity(hash, memory.Hash) - cortex.PositiveCosineSimilarity;
                if (a > 0.0f)
                {
                    positiveActivity += a;
                    positive_MemoryCount += 1;
                }
                else //if (a > -0.66)
                {
                    negativeActivity += a;
                    negative_MemoryCount += 1;
                }

                if (float.IsNaN(positiveActivity) ||
                        float.IsNaN(negativeActivity))
                    throw new Exception();
            }

            if (positive_MemoryCount > 0)
                positiveActivity /= positive_MemoryCount;

            if (negative_MemoryCount > 0)
                negativeActivity /= negative_MemoryCount;

            return (positiveActivity, negativeActivity);
            //return (positive_MemoryCount, -negative_MemoryCount);
        }

        public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
        {
            //float superActivity = Temp_Activity.Item1 + Temp_Activity.Item2;
            float positiveActivitySum = 0.0f;
            float positiveActivitySum_TotalK = 0.0f;

            float negativeActivitySum = 0.0f;
            float negativeActivitySum_TotalK = 0.0f;

            foreach (var r in Enumerable.Range(0, miniColumn.NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = miniColumn.NearestMiniColumnInfos[r];

                int nearestMiniColumnsForRCount = 0;
                float positiveActivitySumForR = 0.0f;
                float negativeActivitySumForR = 0.0f;
                foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
                {
                    var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
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

                float positiveK = cortex.PositiveK[r];
                positiveActivitySum += positiveK * positiveActivitySumForR / nearestMiniColumnsForRCount;
                positiveActivitySum_TotalK += positiveK;

                float negativeK = cortex.NegativeK[r];
                negativeActivitySum += negativeK * negativeActivitySumForR / nearestMiniColumnsForRCount;
                negativeActivitySum_TotalK += negativeK;
            }

            //if (positiveActivitySum_TotalK > 0)
            //    positiveActivitySum /= positiveActivitySum_TotalK;

            //if (negativeActivitySum_TotalK > 0)
            //    negativeActivitySum /= negativeActivitySum_TotalK;

            return positiveActivitySum + negativeActivitySum;            
        }

        public class ActivitiyMaxInfo
        {
            public float MaxActivity = float.MinValue;
            public readonly List<Cortex.MiniColumn> ActivityMax_MiniColumns = new();
            public Cortex.MiniColumn? GetActivityMax_MiniColumn(Random random)
            {
                if (ActivityMax_MiniColumns.Count == 0)
                    return null;
                if (ActivityMax_MiniColumns.Count == 1)
                    return ActivityMax_MiniColumns[0];
                return ActivityMax_MiniColumns[random.Next(ActivityMax_MiniColumns.Count)];
            }

            public float MaxSuperActivity = float.MinValue;
            public readonly List<Cortex.MiniColumn> SuperActivityMax_MiniColumns = new();
            public Cortex.MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
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
