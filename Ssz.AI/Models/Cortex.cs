using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models
{
    public class Cortex
    {
        /// <summary>
        ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
        /// </summary>
        /// <param name="constants"></param>
        /// <param name="detectors"></param>
        public Cortex(
            ICortexConstants constants,            
            Retina retina)
        {
            Constants = constants;

            Random random = new();

            DetectorsVisibleRadius = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.DetectorDelta * constants.DetectorDelta / Math.PI);

            MiniColumns = new MiniColumn[constants.CortexWidth, constants.CortexHeight];
            double minCenterX = DetectorsVisibleRadius;
            double maxCenterX = constants.ImageWidth - DetectorsVisibleRadius;
            double deltaCenterX = (maxCenterX - minCenterX) / (constants.CortexWidth - 1);
            double minCenterY = DetectorsVisibleRadius;
            double maxCenterY = constants.ImageHeight - DetectorsVisibleRadius;
            double deltaCenterY = (maxCenterY - minCenterY) / (constants.CortexHeight - 1);
            
            if (constants.SubAreaMiniColumnsCount is not null)
                SubAreaMiniColumnsRadius = Math.Sqrt(constants.SubAreaMiniColumnsCount.Value / Math.PI);
            else
                SubAreaMiniColumnsRadius = 0.0;            

            Parallel.For(
                fromInclusive: 0,
                toExclusive: constants.CortexHeight,
                mcy =>
                {
                    foreach (int mcx in Enumerable.Range(0, constants.CortexWidth))
                    {
                        double miniColumnR = Math.Sqrt((mcx - constants.SubAreaCenter_Cx) * (mcx - constants.SubAreaCenter_Cx) + (mcy - constants.SubAreaCenter_Cy) * (mcy - constants.SubAreaCenter_Cy));
                        if (SubAreaMiniColumnsRadius == 0.0 || miniColumnR < SubAreaMiniColumnsRadius)
                        {
                            double centerX = minCenterX + mcx * deltaCenterX;
                            double centerY = minCenterY + mcy * deltaCenterY;

                            List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

                            for (int dy = (int)((centerY - DetectorsVisibleRadius) / constants.DetectorDelta); dy < (int)((centerY + DetectorsVisibleRadius) / constants.DetectorDelta) && dy < retina.Detectors.GetLength(1); dy += 1)
                                for (int dx = (int)((centerX - DetectorsVisibleRadius) / constants.DetectorDelta); dx < (int)((centerX + DetectorsVisibleRadius) / constants.DetectorDelta) && dx < retina.Detectors.GetLength(0); dx += 1)
                                {
                                    Detector detector = retina.Detectors[dx, dy];
                                    double r = Math.Sqrt((detector.CenterX - centerX) * (detector.CenterX - centerX) + (detector.CenterY - centerY) * (detector.CenterY - centerY));
                                    if (r < DetectorsVisibleRadius)
                                        miniColumnDetectors.Add(detector);
                                }

                            MiniColumn miniColumn = new MiniColumn(
                                constants,
                                mcx,
                                mcy,
                                miniColumnDetectors,
                                centerX,
                                centerY,                                
                                random);

                            MiniColumns[mcx, mcy] = miniColumn;

                            if (miniColumnR < 0.000001)
                                CenterMiniColumn = miniColumn;
                        }
                    }
                });

            HashSet<Detector> subArea_DetectorsHashSet = new(retina.Detectors.GetLength(0) * retina.Detectors.GetLength(1));
            List<MiniColumn> subArea_MiniColums = new(constants.SubAreaMiniColumnsCount ?? (MiniColumns.GetLength(0) * MiniColumns.GetLength(1)));

            foreach (int mcy in Enumerable.Range(0, MiniColumns.GetLength(1)))
                foreach (int mcx in Enumerable.Range(0, MiniColumns.GetLength(0)))
                {
                    var mc = MiniColumns[mcx, mcy];
                    if (mc is not null)
                    {
                        subArea_MiniColums.Add(mc);
                        foreach (var d in mc.Detectors)
                        {
                            subArea_DetectorsHashSet.Add(d);
                        }
                    }
                }
            SubArea_MiniColumns = subArea_MiniColums.ToArray();
            SubArea_Detectors = subArea_DetectorsHashSet.ToArray();

            // Находим ближайшие миниколонки для каждой миниколонки
            Parallel.For(
                fromInclusive: 0,
                toExclusive: SubArea_MiniColumns.Length,
                mci =>
                {
                    MiniColumn mc = SubArea_MiniColumns[mci];

                    foreach (int r in Enumerable.Range(0, constants.NearestMiniColumnsDelta))
                    {
                        mc.NearestMiniColumnInfos.Add(((2.0f / ((r + 2.0f)), 1.0f / ((r + 2.0f))), new List<MiniColumn>(constants.NearestMiniColumnsDelta * constants.NearestMiniColumnsDelta * 4)));
                    }

                    for (int mcy = mc.MCY - constants.NearestMiniColumnsDelta; mcy < mc.MCY + constants.NearestMiniColumnsDelta; mcy += 1)
                        for (int mcx = mc.MCX - constants.NearestMiniColumnsDelta; mcx < mc.MCX + constants.NearestMiniColumnsDelta; mcx += 1)
                        {
                            if (mcx < 0 ||
                                    mcx >= constants.CortexWidth ||
                                    mcy < 0 ||
                                    mcy >= constants.CortexHeight ||
                                    (mcx == mc.MCX && mcy == mc.MCY))
                                continue;                            

                            MiniColumn nearestMc = MiniColumns[mcx, mcy];
                            if (nearestMc is null)
                                continue;
                            double r = Math.Sqrt((mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY));
                            if (r < constants.NearestMiniColumnsDelta)
                            {
                                mc.NearestMiniColumnInfos[(int)(r - 0.5f)].Item2.Add(nearestMc);
                            }
                        }
                });            

            VisualizationTableItems = new(1000);
        }

        public double SubAreaMiniColumnsRadius;

        /// <summary>
        ///     In big picture coordinates
        /// </summary>
        public double DetectorsVisibleRadius { get; }

        public ICortexConstants Constants { get; }

        public MiniColumn[,] MiniColumns { get; }

        public MiniColumn? CenterMiniColumn { get; private set; }

        public MiniColumn[] SubArea_MiniColumns { get; } = null!;
        public Detector[] SubArea_Detectors { get; } = null!;

        public List<VisualizationTableItem> VisualizationTableItems { get; }
    }

    public class MiniColumn
    {
        public MiniColumn(ICortexConstants constants, int mcx, int mcy, List<Detector> detectors, double centerX, double centerY, Random random)
        {
            Constants = constants;
            Detectors = detectors;
            MCX = mcx;
            MCY = mcy;
            CenterX = centerX;
            CenterY = centerY;
            Temp_Hash = new float[constants.HashLength];
            //var hash0 = new float[constants.HashLength];
            //foreach (var _ in Enumerable.Range(0, constants.InitialMemoryBitsCount))
            //{
            //    hash0[random.Next(hash0.Length)] = 1.0f;
            //}
            Memories = new(constants.MemoriesMaxCount); // { new Memory { Hash = hash0 } };
            Temp_Memories = new(constants.MemoriesMaxCount);

            NearestMiniColumnInfos = new List<((float, float), List<MiniColumn>)>();
        }        

        public readonly ICortexConstants Constants;

        public readonly List<Detector> Detectors;        

        /// <summary>
        ///     Индекс миниколонки в матрице по оси X (горизонтально вправо)
        /// </summary>
        public readonly int MCX;

        /// <summary>
        ///     Индекс миниколонки в матрице по оси Y (вертикально вниз)
        /// </summary>
        public readonly int MCY;
        
        /// <summary>
        ///     (Величина, обратно пропорциональная расстоянию; List MiniColumn)
        /// </summary>
        public readonly List<((float, float), List<MiniColumn>)> NearestMiniColumnInfos;

        /// <summary>
        ///     [0..MNISTImageWidth]
        /// </summary>
        public readonly double CenterX;

        /// <summary>
        ///     [0..MNISTImageHeight]
        /// </summary>
        public readonly double CenterY;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public List<Memory> Memories;

        /// <summary>
        ///     Временный список для сохраненных хэш-кодов
        /// </summary>
        public List<Memory> Temp_Memories;

        /// <summary>
        ///     Текущая активность миниколонки при подаче примера.
        ///     Активность по похожести (положительная величина), активность по непохожести (отрицательная величина).
        /// </summary>
        public (float, float) Temp_Activity;

        /// <summary>
        ///     Текущая суммарная активность миниколонки с учетом активностей соседей при подаче примера
        /// </summary>
        public float Temp_SuperActivity;

        /// <summary>
        ///     Текущий хэш активных детекторов при подаче примера.
        /// </summary>
        public readonly float[] Temp_Hash;

        public Color Temp_ActivityColor;

        public Color Temp_SuperActivityColor;

        /// <summary>
        ///     Возвращает активность по похожести (положительная величина), активность по непохожести (отрицательная величина)
        /// </summary>
        /// <param name="hash"></param>
        /// <returns></returns>
        public (float, float) GetActivity(float[] hash)
        {            
            if (TensorPrimitives.Sum(hash) < Constants.MinBitsInHashForMemory)
                //return (0.0f, 0.0f);
                return (float.NaN, float.NaN);

            float positiveActivity = 0.0f;
            int positive_MemoryCount = 0;

            float negativeActivity = 0.0f;            
            int negative_MemoryCount = 0;

            foreach (var mi in Enumerable.Range(0, Memories.Count))
            {
                var memory = Memories[mi];
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

        public float GetSuperActivity()
        {
            //float superActivity = Temp_Activity.Item1 + Temp_Activity.Item2;
            float positiveActivitySum = Temp_Activity.Item1;
            float positiveActivitySum_TotalK = 1.0f;

            float negativeActivitySum = Temp_Activity.Item2;
            float negativeActivitySum_TotalK = 1.0f;

            foreach (var r in Enumerable.Range(0, NearestMiniColumnInfos.Count))
            {
                var nearestMiniColumnInfosForR = NearestMiniColumnInfos[r];

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

        public void CalculateHash(float[] hash)
        {
            Array.Clear(hash);            

            foreach (var detector in Detectors)
            {
                if (detector.Temp_IsActivated)
                    hash[detector.BitIndexInHash] = 1.0f;
            }
        }        
    }

    public struct Memory
    {
        public float[] Hash;       
        public bool IsDeleted;
    }

    public class ActivitiyMaxInfo
    {        
        public float MaxActivity = float.MinValue;
        public readonly List<MiniColumn> ActivityMax_MiniColumns = new();
        public MiniColumn? GetActivityMax_MiniColumn(Random random)
        {
            if (ActivityMax_MiniColumns.Count == 0)
                return null;
            if (ActivityMax_MiniColumns.Count == 1)
                return ActivityMax_MiniColumns[0];
            return ActivityMax_MiniColumns[random.Next(ActivityMax_MiniColumns.Count)];
        }

        public float MaxSuperActivity = float.MinValue;
        public readonly List<MiniColumn> SuperActivityMax_MiniColumns = new();
        public MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
        {
            if (SuperActivityMax_MiniColumns.Count == 0)
                return null;
            if (SuperActivityMax_MiniColumns.Count == 1)
                return SuperActivityMax_MiniColumns[0];
            return SuperActivityMax_MiniColumns[random.Next(SuperActivityMax_MiniColumns.Count)];
        }
    }    

    public interface ICortexConstants
    {
        /// <summary>
        ///     Количество миниколонок в зоне коры по оси X
        /// </summary>
        int CortexWidth { get; }

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси Y
        /// </summary>
        int CortexHeight { get; }

        /// <summary>
        ///     Ширина основного изображения
        /// </summary>
        int ImageWidth { get; }

        /// <summary>
        ///     Высота основного изображения
        /// </summary>
        int ImageHeight { get; }

        /// <summary>
        ///     Количество детекторов, видимых одной миниколонкой
        /// </summary>
        int MiniColumnVisibleDetectorsCount { get; }

        /// <summary>
        ///     Расстояние между детекторами по коризонтали и вертикали  
        /// </summary>
        double DetectorDelta { get; }    
        
        /// <summary>
        ///     Количество миниколонок в подобласти
        /// </summary>
        int? SubAreaMiniColumnsCount { get; }

        /// <summary>
        ///     Индекс X центра подобласти
        /// </summary>
        int SubAreaCenter_Cx { get; }

        /// <summary>
        ///     Индекс Y центра подобласти
        /// </summary>
        int SubAreaCenter_Cy { get; }

        /// <summary>
        ///     Длина хэш-вектора
        /// </summary>
        int HashLength { get; }

        /// <summary>
        ///     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
        /// </summary>
        int InitialMemoryBitsCount { get; }

        /// <summary>
        ///     Минимальное число бит в хэше, что бы быть сохраненным в память
        /// </summary>
        int MinBitsInHashForMemory { get; }

        /// <summary>
        ///     Максимальное расстояние до ближайших миниколонок
        /// </summary>
        int NearestMiniColumnsDelta { get; }                

        /// <summary>
        ///     Верхний предел количества воспоминаний (для кэширования)
        /// </summary>
        int MemoriesMaxCount { get; }
    }
}
