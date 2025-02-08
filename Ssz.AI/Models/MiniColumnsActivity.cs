using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models
{
    public static class MiniColumnsActivity
    {
        /// <summary>
        ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина), общая активность
        /// </summary>
        /// <param name="hash"></param>
        /// <returns></returns>
        public static (float, float, float) GetActivity(Cortex.MiniColumn miniColumn, float[] hash, Cortex cortex)
        {
            if (TensorPrimitives.Sum(hash) < miniColumn.Constants.MinBitsInHashForMemory)
                //return (0.0f, 0.0f);
                return (float.NaN, float.NaN, float.NaN);            

            float positiveActivity = 0.0f;
            int positive_MemoryCount = 0;           

            float negativeActivity = 0.0f;
            int negative_MemoryCount = 0;

            float activity = 0.0f;
            int memoryCount = 0;

            foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
            {
                var memory = miniColumn.Memories[mi];
                if (memory.IsDeleted)
                    continue;

                float a = TensorPrimitives.CosineSimilarity(hash, memory.Hash);
                if (float.IsNaN(a))
                    throw new Exception();

                activity += a;
                memoryCount += 1;

                a -= cortex.PositiveCosineSimilarity;
                if (a > 0.0f)
                {
                    positiveActivity += a;
                    positive_MemoryCount += 1;                    
                }
                else
                {
                    negativeActivity += a;
                    negative_MemoryCount += 1;                    
                }
            }
            
            if (memoryCount > 0)
                activity /= memoryCount;
            activity -= cortex.PositiveCosineSimilarity;

            if (positive_MemoryCount > 0)
                positiveActivity /= positive_MemoryCount;

            if (negative_MemoryCount > 0)
                negativeActivity /= negative_MemoryCount;

            return (positiveActivity, negativeActivity, activity);
            //return (positive_MemoryCount, -negative_MemoryCount);
        }

        public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
        {
            if (float.IsNaN(miniColumn.Temp_Activity.Item3))
                return float.NaN;            

            float positiveActivitySum = 0.0f;
            float positiveActivitySum_TotalK = 0.0f;

            float negativeActivitySum = 0.0f;
            float negativeActivitySum_TotalK = 0.0f;

            float activitySum = 0.0f;
            //if (miniColumn.Temp_Activity.Item3 + cortex.PositiveCosineSimilarity > cortex.PositiveCosineSimilarity2)
            //{
            //    activitySum = 1.0f;
            //}
            if (miniColumn.Memories.Count(m => !m.IsDeleted) == 0)
            {
                activitySum = 1.0f;
            }

            foreach (var r in Enumerable.Range(0, miniColumn.NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = miniColumn.NearestMiniColumnInfos[r];

                int nearestMiniColumnsForRCount = 0;
                float positiveActivitySumForR = 0.0f;
                float negativeActivitySumForR = 0.0f;
                float activitySumForR = 0.0f;
                foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
                {
                    var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
                    if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item3))
                    {
                        positiveActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item1;
                        negativeActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item2;
                        activitySumForR += nearestMiniColumnInfo.Temp_Activity.Item3; // * Math.Abs(nearestMiniColumnInfo.Temp_Activity.Item3);
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

                float activityForR = activitySumForR / nearestMiniColumnsForRCount;
                //if (r == 0)
                //    activityForR *= Math.Abs(activityForR);

                activitySum += positiveK * activityForR;
            }

            //if (positiveActivitySum_TotalK > 0)
            //    positiveActivitySum /= positiveActivitySum_TotalK;

            //if (negativeActivitySum_TotalK > 0)
            //    negativeActivitySum /= negativeActivitySum_TotalK;
            //if (float.IsNaN(positiveActivitySum) || float.IsNaN(negativeActivitySum))
            //{
            //}

            //return positiveActivitySum + negativeActivitySum;            
            return activitySum; // * Math.Abs(activitySum);
        }        
    }

    public static class MiniColumnsActivity2
    {
        /// <summary>
        ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина), общая активность
        /// </summary>
        /// <param name="hash"></param>
        /// <returns></returns>
        public static (float, float, float) GetActivity(Cortex.MiniColumn miniColumn, float[] hash, Cortex cortex)
        {
            if (TensorPrimitives.Sum(hash) < miniColumn.Constants.MinBitsInHashForMemory)                
                return (float.NaN, float.NaN, float.NaN);

            float positiveActivity = 0.0f;            

            float negativeActivity = 0.0f;
            
            foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
            {
                var memory = miniColumn.Memories[mi];
                if (memory.IsDeleted)
                    continue;

                float a = TensorPrimitives.CosineSimilarity(hash, memory.Hash);
                if (float.IsNaN(a))
                    throw new Exception();
                
                a -= cortex.PositiveCosineSimilarity;
                if (a > 0.0f)
                {
                    positiveActivity += a;                    
                }
                else
                {
                    negativeActivity += a;                    
                }
            }           

            return (positiveActivity, negativeActivity, positiveActivity + negativeActivity);           
        }

        public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
        {
            if (float.IsNaN(miniColumn.Temp_Activity.Item3))
                return float.NaN;

            int count = 0;
            float activitySum = 0.0f;            
            if (miniColumn.Memories.Count(m => !m.IsDeleted) == 0)
            {
                activitySum = 1000.0f;
            }

            foreach (var r in Enumerable.Range(0, miniColumn.NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = miniColumn.NearestMiniColumnInfos[r];
                float positiveK = cortex.PositiveK[r];

                foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
                {
                    var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
                    if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item3))
                    {                        
                        activitySum += positiveK * nearestMiniColumnInfo.Temp_Activity.Item3; // * Math.Abs(nearestMiniColumnInfo.Temp_Activity.Item3);
                        count += 1;
                    }
                }
            }

            if (count > 0)
                activitySum /= count;

            return activitySum;
        }
    }

    public class ActivitiyMaxInfo
    {
        public float MaxActivity = float.MinValue;
        public readonly List<Cortex.MiniColumn> ActivityMax_MiniColumns = new();

        public float MaxSuperActivity = float.MinValue;
        public readonly List<Cortex.MiniColumn> SuperActivityMax_MiniColumns = new();
        public Cortex.MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
        {
            if (SuperActivityMax_MiniColumns.Count == 0)
                return null;
            if (SuperActivityMax_MiniColumns.Count == 1)
                return SuperActivityMax_MiniColumns[0];
            var winnerIndex = random.Next(SuperActivityMax_MiniColumns.Count);
            return SuperActivityMax_MiniColumns[winnerIndex];
        }
    }
}
