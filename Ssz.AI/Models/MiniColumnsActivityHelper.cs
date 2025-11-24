using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models;

public interface IMiniColumnsActivityConstants
{
    int DiscreteVectorLength { get; }

    int DiscreteOptimizedVector_PrimaryBitsCount { get; }

    /// <summary>
    ///     Количество миниколонок в зоне коры по оси X
    /// </summary>
    int CortexWidth_MiniColumns { get; }

    /// <summary>
    ///     Количество миниколонок в зоне коры по оси Y
    /// </summary>
    int CortexHeight_MiniColumns { get; }

    /// <summary>
    ///     Нулевой уровень косинусного подобия
    /// </summary>
    float K0 { get; set; }

    /// <summary>
    ///     Косинусное подобие с пустой миниколонкой
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

public interface IMiniColumn
{
    int MCX { get; }
    int MCY { get; }

    IFastList<ICortexMemory?> CortexMemories { get; }

    int DiscreteOptimizedVector_ProjectionIndex { get; }
}

public interface IMiniColumnActivity
{
    IMiniColumn MiniColumn { get; }

    (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Activity { get; set; }

    float SuperActivity { get; set; }

    IFastList<(float, float, IMiniColumnActivity)> K_ForNearestMiniColumns { get; }
}

public interface ICortexMemory
{
    float[] DiscreteVector { get; }
}

public class ActivitiyMaxInfo
{
    public IMiniColumn? SelectedSuperActivityMax_MiniColumn;

    public float MaxActivity = float.MinValue;
    public readonly List<IMiniColumn> ActivityMax_MiniColumns = new();

    public float MaxSuperActivity = float.MinValue;
    public readonly List<IMiniColumn> SuperActivityMax_MiniColumns = new();

    public IMiniColumn? GetSuperActivityMax_MiniColumn(Random random)
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
    public static (float, float, int) GetActivity(IMiniColumn miniColumn, float[] discreteVector, IMiniColumnsActivityConstants constants)
    {
        float positiveActivity = 0.0f;
        int positiveMemoriesCount = 0;
        float negativeActivity = 0.0f;
        int negativeMemoriesCount = 0;

        foreach (var mi in Enumerable.Range(0, miniColumn.CortexMemories.Count))
        {
            var cortexMemories = miniColumn.CortexMemories[mi];
            if (cortexMemories is null)
                continue;

            float cosineSimilarity = TensorPrimitives.CosineSimilarity(discreteVector, cortexMemories.DiscreteVector);
            //if (memoryCosineSimilarity > constants.K1)
            {
                float activity = cosineSimilarity - constants.K0;
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

    /// <summary>
    ///     
    /// </summary>
    /// <param name="miniColumnActivity"></param>
    /// <param name="constants"></param>
    /// <returns></returns>
    public static float GetSuperActivity(IMiniColumnActivity miniColumnActivity, IMiniColumnsActivityConstants constants)
    {
        float superActivity;

        var k0 = miniColumnActivity.K_ForNearestMiniColumns[0];
        if (miniColumnActivity.Activity.CortexMemoriesCount > 0)
            superActivity = k0.Item1 * miniColumnActivity.Activity.PositiveActivity + k0.Item2 * miniColumnActivity.Activity.NegativeActivity;
        else
            superActivity = k0.Item1 * (constants.K2 - constants.K0); // Best proximity

        for (int i = 1; i < miniColumnActivity.K_ForNearestMiniColumns.Count; i += 1)
        {
            var it = miniColumnActivity.K_ForNearestMiniColumns[i];
            var nearestMiniColumn = it.Item3;

            if (nearestMiniColumn.Activity.CortexMemoriesCount > 0)
                superActivity += it.Item1 * nearestMiniColumn.Activity.PositiveActivity +
                    it.Item2 * nearestMiniColumn.Activity.NegativeActivity;
            //else
            //    superActivity += it.Item1 * (constants.K2 - constants.K0); // Best proximity
        }

        return superActivity;
    }

    //public static void GetSuperActivity(IMiniColumn[] miniColumns, int bitsCount, IMiniColumnsActivityConstants constants, float[] result)
    //{
    //    Array.Clear(result);

    //    for (int i = 0; i < miniColumns.Length; i += 1)
    //    {
    //        var miniColumn = miniColumns[i];

    //        for (int j = 0; j < miniColumn.CortexMemories.Count; j += 1)
    //        {
    //            var cortexMemory = miniColumn.CortexMemories[j];
    //            if (cortexMemory is not null)
    //                TensorPrimitives.Add(result, cortexMemory.DiscreteVector, result);
    //        }
    //    }
    //}

    public static void CalculateActivityAndSuperActivity(float[] discreteVector, IDenseMatrix<IMiniColumnActivity> miniColumnActivities, ActivitiyMaxInfo? activitiyMaxInfo, IMiniColumnsActivityConstants constants)
    {
        for (int mci = 0; mci < miniColumnActivities.Data.Length; mci += 1)
        {
            var miniColumnActivity = miniColumnActivities.Data[mci];
            miniColumnActivity.Activity = GetActivity(miniColumnActivity.MiniColumn, discreteVector, constants);
        };

        if (activitiyMaxInfo is not null)
        {
            activitiyMaxInfo.MaxActivity = float.MinValue;
            activitiyMaxInfo.ActivityMax_MiniColumns.Clear();

            //if (Constants.SuperactivityThreshold)
            //    activitiyMaxInfo.MaxSuperActivity = Constants.K4;
            //else
            activitiyMaxInfo.MaxSuperActivity = float.MinValue;
            activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();

            for (int mci = 0; mci < miniColumnActivities.Data.Length; mci += 1)
            {
                var miniColumnActivity = miniColumnActivities.Data[mci];
                miniColumnActivity.SuperActivity = GetSuperActivity(miniColumnActivity, constants);

                float a = miniColumnActivity.Activity.PositiveActivity + miniColumnActivity.Activity.NegativeActivity;
                if (a > activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.MaxActivity = a;
                    activitiyMaxInfo.ActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }
                else if (a == activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }

                if (miniColumnActivity.SuperActivity > activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.MaxSuperActivity = miniColumnActivity.SuperActivity;
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }
                else if (miniColumnActivity.SuperActivity == activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }
            }
        }
        else
        {
            for (int mci = 0; mci < miniColumnActivities.Data.Length; mci += 1)
            {
                var miniColumnActivity = miniColumnActivities.Data[mci];
                miniColumnActivity.SuperActivity = GetSuperActivity(miniColumnActivity, constants);
            }
        }   
    }

    public static void ReCalculateActivityAndSuperActivity(
        float[] discreteVector, 
        IDenseMatrix<IMiniColumnActivity> miniColumnActivities, 
        IMiniColumnsActivityConstants constants, 
        IMiniColumn prevIteratorMiniColumn, 
        IMiniColumn iteratorMiniColumn)
    {
        var miniColumnActivity = miniColumnActivities[prevIteratorMiniColumn.MCX, prevIteratorMiniColumn.MCY];
        miniColumnActivity.Activity = GetActivity(prevIteratorMiniColumn, discreteVector, constants);

        miniColumnActivity = miniColumnActivities[iteratorMiniColumn.MCX, iteratorMiniColumn.MCY];
        miniColumnActivity.Activity = GetActivity(iteratorMiniColumn, discreteVector, constants);

        int maxR = constants.PositiveK.Length - 1;        

        for (int mcy = Math.Max(0, Math.Min(prevIteratorMiniColumn.MCY, iteratorMiniColumn.MCY) - maxR); 
                mcy < Math.Min(miniColumnActivities.Dimensions[1], Math.Max(prevIteratorMiniColumn.MCY, iteratorMiniColumn.MCY) + maxR + 1); mcy += 1)
            for (int mcx = Math.Max(0, Math.Min(prevIteratorMiniColumn.MCX, iteratorMiniColumn.MCX) - maxR);
                mcx < Math.Min(miniColumnActivities.Dimensions[0], Math.Max(prevIteratorMiniColumn.MCX, iteratorMiniColumn.MCX) + maxR + 1); mcx += 1)
            {
                miniColumnActivity = miniColumnActivities[mcx, mcy];
                miniColumnActivity.SuperActivity = GetSuperActivity(miniColumnActivity, constants);
            }        
    }

    public static void ClearActivityAndSuperActivity(IDenseMatrix<IMiniColumnActivity> miniColumnActivities, ActivitiyMaxInfo? activitiyMaxInfo, IMiniColumnsActivityConstants constants)
    {
        for (int mci = 0; mci < miniColumnActivities.Data.Length; mci += 1)
        {
            var miniColumnActivity = miniColumnActivities.Data[mci];
            miniColumnActivity.Activity = (0.0f, 0.0f, 0);
            miniColumnActivity.SuperActivity = GetSuperActivity(miniColumnActivity, constants);
        };

        if (activitiyMaxInfo is not null)
        {
            activitiyMaxInfo.MaxActivity = float.MinValue;
            activitiyMaxInfo.ActivityMax_MiniColumns.Clear();

            //if (Constants.SuperactivityThreshold)
            //    activitiyMaxInfo.MaxSuperActivity = Constants.K4;
            //else
            activitiyMaxInfo.MaxSuperActivity = float.MinValue;
            activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();

            for (int mci = 0; mci < miniColumnActivities.Data.Length; mci += 1)
            {
                var miniColumnActivity = miniColumnActivities.Data[mci];
                miniColumnActivity.SuperActivity = GetSuperActivity(miniColumnActivity, constants);

                float a = miniColumnActivity.Activity.PositiveActivity + miniColumnActivity.Activity.NegativeActivity;
                if (a > activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.MaxActivity = a;
                    activitiyMaxInfo.ActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }
                else if (a == activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }

                if (miniColumnActivity.SuperActivity > activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.MaxSuperActivity = miniColumnActivity.SuperActivity;
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }
                else if (miniColumnActivity.SuperActivity == activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(miniColumnActivity.MiniColumn);
                }
            }
        }
        else
        {
            for (int mci = 0; mci < miniColumnActivities.Data.Length; mci += 1)
            {
                var miniColumnActivity = miniColumnActivities.Data[mci];
                miniColumnActivity.SuperActivity = GetSuperActivity(miniColumnActivity, constants);
            }
        }
    }    
}        


//public static class MiniColumnsActivity_Old
//{
//    /// <summary>
//    ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина), общая активность
//    /// </summary>
//    /// <param name="hash"></param>
//    /// <returns></returns>
//    public static (float, float, float) GetActivity(Cortex.MiniColumn miniColumn, float[] hash, Cortex cortex)
//    {
//        if (TensorPrimitives.Sum(hash) < miniColumn.Constants.MinBitsInHashForMemory)
//            //return (0.0f, 0.0f);
//            return (float.NaN, float.NaN, float.NaN);

//        float positiveActivity = 0.0f;
//        //int positive_MemoriesCount = 0;           

//        float negativeActivity = 0.0f;
//        //int negative_MemoriesCount = 0;

//        float cosineSimilaritySum = 0.0f;
//        //int memoriesCount = 0;

//        foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
//        {
//            var memory = miniColumn.Memories[mi];
//            if (memory.IsDeleted)
//                continue;

//            float cosineSimilarity = TensorPrimitives.CosineSimilarity(hash, memory.Hash);
//            if (float.IsNaN(cosineSimilarity))
//                throw new Exception();

//            cosineSimilaritySum += cosineSimilarity;
//            //memoriesCount += 1;

//            float a = cosineSimilarity - cortex.K0;
//            if (a > 0.0f)
//            {
//                positiveActivity += a;
//                //positive_MemoriesCount += 1;                    
//            }
//            else
//            {
//                negativeActivity += a;
//                //negative_MemoriesCount += 1;                    
//            }
//        }

//        //if (positive_MemoriesCount > 0)
//        //    positiveActivity = positiveActivity / positive_MemoriesCount;

//        //float activity;
//        //if (memoriesCount > 0)
//        //    activity = cosineSimilaritySum / memoriesCount - cortex.PositiveCosineSimilarity;
//        //else
//        //    activity = 0.0f;

//        return (positiveActivity, negativeActivity, cosineSimilaritySum);
//    }

//    public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
//    {
//        if (float.IsNaN(miniColumn.Temp_Activity.Item3))
//            return float.NaN;

//        float superActivity = 0.0f; // miniColumn.Temp_Activity.Item2;
//                                    //int negativeMemoriesCount = 0;
//        foreach (var memory in miniColumn.Memories.Where(m => !m.IsDeleted))
//        {
//            superActivity -= cortex.K1;
//            //float memoryActivity = (memory.Temp_CosineSimilarity - cortex.PositiveCosineSimilarity2);
//            //if (memoryActivity < 0.0f)
//            //{
//            //    superActivity += memoryActivity;
//            //    //negativeMemoriesCount += 1;
//            //}
//        }
//        //if (negativeMemoriesCount > 0)
//        //    superActivity /= negativeMemoriesCount;

//        foreach (var r in Enumerable.Range(0, miniColumn.K_ForNearestMiniColumns.Count))
//        {
//            var nearestMiniColumnInfosForR = miniColumn.K_ForNearestMiniColumns[r];

//            //float positiveActivitySumForR = 0.0f;
//            //float negativeActivitySumForR = 0.0f;
//            float activitySumForR = 0.0f;
//            int nearestMiniColumnsForRCount = 0;
//            foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
//            {
//                var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
//                if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item3))
//                {
//                    //positiveActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item1;
//                    //negativeActivitySumForR += nearestMiniColumnInfo.Temp_Activity.Item2;
//                    activitySumForR += nearestMiniColumnInfo.Temp_Activity.Item3;
//                    nearestMiniColumnsForRCount += 1;
//                }
//            }

//            if (nearestMiniColumnsForRCount == 0)
//                continue;

//            float positiveK = cortex.PositiveK[r];
//            //float negativeK = cortex.NegativeK[r];

//            superActivity += positiveK * activitySumForR;
//        }

//        return superActivity;
//    }
//}

//public static class MiniColumnsActivity_Old2
//{
//    /// <summary>
//    ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина), общая активность
//    /// </summary>
//    /// <param name="hash"></param>
//    /// <returns></returns>
//    public static (float, float, float) GetActivity(Cortex.MiniColumn miniColumn, float[] hash, Cortex cortex)
//    {
//        if (TensorPrimitives.Sum(hash) < miniColumn.Constants.MinBitsInHashForMemory)
//            return (float.NaN, float.NaN, float.NaN);

//        float positiveActivity = 0.0f;

//        float negativeActivity = 0.0f;

//        foreach (var mi in Enumerable.Range(0, miniColumn.Memories.Count))
//        {
//            var memory = miniColumn.Memories[mi];
//            if (memory.IsDeleted)
//                continue;

//            float a = TensorPrimitives.CosineSimilarity(hash, memory.Hash);
//            if (float.IsNaN(a))
//                throw new Exception();

//            a -= cortex.K0;
//            if (a > 0.0f)
//            {
//                positiveActivity += a;
//            }
//            else
//            {
//                negativeActivity += a;
//            }
//        }

//        return (positiveActivity, negativeActivity, positiveActivity + negativeActivity);
//    }

//    public static float GetSuperActivity(Cortex.MiniColumn miniColumn, Cortex cortex)
//    {
//        if (float.IsNaN(miniColumn.Temp_Activity.Item3))
//            return float.NaN;

//        int count = 0;
//        float activitySum = 0.0f;
//        if (miniColumn.Memories.Count(m => !m.IsDeleted) == 0)
//        {
//            activitySum = 1000.0f;
//        }

//        foreach (var r in Enumerable.Range(0, miniColumn.K_ForNearestMiniColumns.Count))
//        {
//            var nearestMiniColumnInfosForR = miniColumn.K_ForNearestMiniColumns[r];
//            float positiveK = cortex.PositiveK[r];

//            foreach (var mci in Enumerable.Range(0, nearestMiniColumnInfosForR.Count))
//            {
//                var nearestMiniColumnInfo = nearestMiniColumnInfosForR[mci];
//                if (!float.IsNaN(nearestMiniColumnInfo.Temp_Activity.Item3))
//                {
//                    activitySum += positiveK * nearestMiniColumnInfo.Temp_Activity.Item3; // * Math.Abs(nearestMiniColumnInfo.Temp_Activity.Item3);
//                    count += 1;
//                }
//            }
//        }

//        if (count > 0)
//            activitySum /= count;

//        return activitySum;
//    }
//}