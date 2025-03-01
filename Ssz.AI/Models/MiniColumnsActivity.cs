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
                return (float.NaN, float.NaN, float.NaN);

            float positiveActivity = 0.0f;
            int positive_MemoriesCount = 0;           

            float negativeActivity = 0.0f;
            int negative_MemoriesCount = 0;

            float activity = 0.0f;
            int memoriesCount = 0;

            foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
            {
                var memory = miniColumn.Memories[mi];
                if (memory.IsDeleted)
                    continue;

                float memoryCosineSimilarity = TensorPrimitives.CosineSimilarity(hash, memory.Hash);
                if (float.IsNaN(memoryCosineSimilarity))
                    throw new Exception();

                float memoryActivity = memoryCosineSimilarity - cortex.K0;                               

                activity += memoryActivity;
                memoriesCount += 1;

                if (memoryCosineSimilarity >= cortex.K1)
                {
                    positiveActivity += memoryActivity;
                    positive_MemoriesCount += 1;
                }

                //float memoryNegativeActivity = memoryCosineSimilarity - 1.0f;
                //negativeActivity += memoryNegativeActivity * 10; 
                //negative_MemoriesCount += 1;
                //float memoryNegativeActivity = memoryCosineSimilarity - cortex.K2;
                if (memoryCosineSimilarity < cortex.K2)
                {
                    negativeActivity += -cortex.K3;                    
                    negative_MemoriesCount += 1;
                }
            }

            if (memoriesCount > 0)
            {
                activity /= memoriesCount;

                positiveActivity /= memoriesCount;
                negativeActivity /= memoriesCount;
            }

            //if (positive_MemoriesCount > 0)
            //    positiveActivity /= positive_MemoriesCount;

            //if (negative_MemoriesCount > 0)
            //    negativeActivity /= negative_MemoriesCount;

            return (positiveActivity, negativeActivity, activity);
        }

        public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
        {
            if (float.IsNaN(miniColumn.Temp_Activity.Item3))
                return float.NaN;

            float superActivity = miniColumn.Temp_Activity.Item2;

            foreach (var r in Enumerable.Range(0, miniColumn.NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = miniColumn.NearestMiniColumnInfos[r];
                
                float activitySumForR = 0.0f;
                int nearestMiniColumnsForRCount = 0;
                foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
                {
                    var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
                    if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item3))
                    {                        
                        activitySumForR += nearestMiniColumnInfo.Temp_Activity.Item1;
                        nearestMiniColumnsForRCount += 1;
                    }
                }

                if (nearestMiniColumnsForRCount == 0)
                    continue;

                float positiveK = cortex.PositiveK[r];                

                superActivity += positiveK * activitySumForR;
            }

            return superActivity;
        }
    }

    public static class MiniColumnsActivity_Old
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
            //int positive_MemoriesCount = 0;           

            float negativeActivity = 0.0f;
            //int negative_MemoriesCount = 0;

            float cosineSimilaritySum = 0.0f;
            //int memoriesCount = 0;

            foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
            {
                var memory = miniColumn.Memories[mi];
                if (memory.IsDeleted)
                    continue;

                float cosineSimilarity = TensorPrimitives.CosineSimilarity(hash, memory.Hash);                
                if (float.IsNaN(cosineSimilarity))
                    throw new Exception();

                cosineSimilaritySum += cosineSimilarity;
                //memoriesCount += 1;

                float a = cosineSimilarity - cortex.K0;
                if (a > 0.0f)
                {
                    positiveActivity += a;
                    //positive_MemoriesCount += 1;                    
                }
                else
                {
                    negativeActivity += a;
                    //negative_MemoriesCount += 1;                    
                }
            }

            //if (positive_MemoriesCount > 0)
            //    positiveActivity = positiveActivity / positive_MemoriesCount;

            //float activity;
            //if (memoriesCount > 0)
            //    activity = cosineSimilaritySum / memoriesCount - cortex.PositiveCosineSimilarity;
            //else
            //    activity = 0.0f;

            return (positiveActivity, negativeActivity, cosineSimilaritySum);            
        }

        public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
        {
            if (float.IsNaN(miniColumn.Temp_Activity.Item3))
                return float.NaN;

            float superActivity = 0.0f; // miniColumn.Temp_Activity.Item2;
            //int negativeMemoriesCount = 0;
            foreach (var memory in miniColumn.Memories.Where(m => !m.IsDeleted))
            {
                superActivity -= cortex.K1;
                //float memoryActivity = (memory.Temp_CosineSimilarity - cortex.PositiveCosineSimilarity2);
                //if (memoryActivity < 0.0f)
                //{
                //    superActivity += memoryActivity;
                //    //negativeMemoriesCount += 1;
                //}
            }
            //if (negativeMemoriesCount > 0)
            //    superActivity /= negativeMemoriesCount;

            foreach (var r in Enumerable.Range(0, miniColumn.NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = miniColumn.NearestMiniColumnInfos[r];
                
                //float positiveActivitySumForR = 0.0f;
                //float negativeActivitySumForR = 0.0f;
                float activitySumForR = 0.0f;
                int nearestMiniColumnsForRCount = 0;
                foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
                {
                    var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
                    if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item3))
                    {
                        //positiveActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item1;
                        //negativeActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item2;
                        activitySumForR += nearestMiniColumnInfo.Temp_Activity.Item3;
                        nearestMiniColumnsForRCount += 1;
                    }
                }

                if (nearestMiniColumnsForRCount == 0)
                    continue;

                float positiveK = cortex.PositiveK[r];
                //float negativeK = cortex.NegativeK[r];

                superActivity += positiveK * activitySumForR;
            }               

            return superActivity;
        }        
    }

    public static class MiniColumnsActivity_Old2
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
                
                a -= cortex.K0;
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
