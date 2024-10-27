using System;
using System.Collections.Generic;
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

            double detectorsVisibleRadius = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.DetectorDelta * constants.DetectorDelta / Math.PI);

            MiniColumns = new MiniColumn[constants.CortexWidth, constants.CortexHeight];
            double minCenterX = detectorsVisibleRadius;
            double maxCenterX = constants.ImageWidth - detectorsVisibleRadius;
            double deltaCenterX = (maxCenterX - minCenterX) / (constants.CortexWidth - 1);
            double minCenterY = detectorsVisibleRadius;
            double maxCenterY = constants.ImageHeight - detectorsVisibleRadius;
            double deltaCenterY = (maxCenterY - minCenterY) / (constants.CortexHeight - 1);


            double subAreaMiniColumnsRadius;
            if (constants.SubAreaMiniColumnsCount is not null)
                subAreaMiniColumnsRadius = Math.Sqrt(constants.SubAreaMiniColumnsCount.Value / Math.PI);
            else
                subAreaMiniColumnsRadius = 0.0;            

            Parallel.For(
                fromInclusive: 0,
                toExclusive: constants.CortexHeight,
                mcy =>
                {
                    foreach (int mcx in Enumerable.Range(0, constants.CortexWidth))
                    {
                        double miniColumnR = Math.Sqrt(Math.Pow(mcx - constants.SubAreaCenter_Cx, 2) + Math.Pow(mcy - constants.SubAreaCenter_Cy, 2));
                        if (subAreaMiniColumnsRadius == 0.0 || miniColumnR < subAreaMiniColumnsRadius)
                        {
                            double centerX = minCenterX + mcx * deltaCenterX;
                            double centerY = minCenterY + mcy * deltaCenterY;

                            List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

                            for (int dy = (int)((centerY - detectorsVisibleRadius) / constants.DetectorDelta); dy < (int)((centerY + detectorsVisibleRadius) / constants.DetectorDelta); dy += 1)
                                for (int dx = (int)((centerX - detectorsVisibleRadius) / constants.DetectorDelta); dx < (int)((centerX + detectorsVisibleRadius) / constants.DetectorDelta); dx += 1)
                                {
                                    Detector detector = retina.Detectors[dx, dy];
                                    double r = Math.Sqrt(Math.Pow(detector.CenterX - centerX, 2) + Math.Pow(detector.CenterY - centerY, 2));
                                    if (r < detectorsVisibleRadius)
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
                        mc.NearestMiniColumnInfos.Add((1.0f / (2 * (r + 1)), new List<MiniColumn>(constants.NearestMiniColumnsDelta * constants.NearestMiniColumnsDelta * 4)));
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
                            double r = Math.Sqrt(Math.Pow(mcx - mc.MCX, 2) + Math.Pow(mcy - mc.MCY, 2));
                            if (r < constants.NearestMiniColumnsDelta)
                            {
                                mc.NearestMiniColumnInfos[(int)(r - 0.5f)].Item2.Add(nearestMc);
                            }
                        }
                });

            // TEMPCODE
            //foreach (int mci in Enumerable.Range(0, SubArea_MiniColumns.Length))
            //{
            //    MiniColumn mc = SubArea_MiniColumns[mci];

            //    foreach (int r in Enumerable.Range(0, constants.NearestMiniColumnsDelta))
            //    {
            //        mc.NearestMiniColumnInfos.Add((1.0f / (2 * (r + 1)), new List<MiniColumn>(constants.NearestMiniColumnsDelta * constants.NearestMiniColumnsDelta * 4)));
            //    }

            //    for (int mcy = mc.MCY - constants.NearestMiniColumnsDelta; mcy < mc.MCY + constants.NearestMiniColumnsDelta; mcy += 1)
            //        for (int mcx = mc.MCX - constants.NearestMiniColumnsDelta; mcx < mc.MCX + constants.NearestMiniColumnsDelta; mcx += 1)
            //        {
            //            if (mcx < 0 ||                                
            //                    mcx >= constants.CortexWidth ||
            //                    mcy < 0 ||                                
            //                    mcy >= constants.CortexHeight ||
            //                    (mcx == mc.MCX && mcy == mc.MCY))
            //                continue;

            //            MiniColumn nearestMc = MiniColumns[mcx, mcy];
            //            if (nearestMc is null)
            //                continue;
            //            double r = Math.Sqrt(Math.Pow(mcx - mc.MCX, 2) + Math.Pow(mcy - mc.MCY, 2));
            //            if (r < constants.NearestMiniColumnsDelta)
            //            {
            //                mc.NearestMiniColumnInfos[(int)(r - 0.5f)].Item2.Add(nearestMc);
            //            }
            //        }
            //}
        }

        public ICortexConstants Constants { get; }

        public MiniColumn[,] MiniColumns { get; }

        public MiniColumn? CenterMiniColumn { get; private set; }

        public MiniColumn[] SubArea_MiniColumns { get; } = null!;
        public Detector[] SubArea_Detectors { get; } = null!;
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
            var hash0 = new float[constants.HashLength];
            foreach (var _ in Enumerable.Range(0, constants.InitialMemoryBitsCount))
            {
                hash0[random.Next(hash0.Length)] = 1.0f;
            }
            Memories = new(constants.MemoriesMaxCount) { new Memory { Hash = hash0 } };

            NearestMiniColumnInfos = new List<(float, List<MiniColumn>)>();
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
        public readonly List<(float, List<MiniColumn>)> NearestMiniColumnInfos;

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
        public readonly List<Memory> Memories;

        /// <summary>
        ///     Текущая активность миниколонки при подаче примера
        /// </summary>
        public float Temp_Activity;

        /// <summary>
        ///     Текущая суммарная активность миниколонки с учетом активностей соседей при подаче примера
        /// </summary>
        public float Temp_SuperActivity;

        /// <summary>
        ///     Текущий хэш активных детекторов при подаче примера.
        /// </summary>
        public readonly float[] Temp_Hash;

        public float GetActivity(float[] hash)
        {
            if (TensorPrimitives.Sum(hash) < Constants.MinBitsInHashForMemory)
                return -Constants.MiniColumnMinimumActivity;

            float activity = 0.0f;

            foreach (var mi in Enumerable.Range(0, Memories.Count))
            {
                var memory = Memories[mi];
                if (memory.IsDeleted)
                    continue;
                activity += TensorPrimitives.CosineSimilarity(hash, memory.Hash) - Constants.MiniColumnMinimumActivity;
            }            

            return activity;
        }        

        public float GetSuperActivity()
        {
            float superActivity = Temp_Activity;
            
            foreach (var r in Enumerable.Range(0, NearestMiniColumnInfos.Count))
            {
                if (NearestMiniColumnInfos[r].Item2.Count == 0)
                    continue;

                float localSuperActivity = 0.0f;
                foreach (var mci in Enumerable.Range(0, NearestMiniColumnInfos[r].Item2.Count))
                {
                    var nearestMiniColumnInfo = NearestMiniColumnInfos[r].Item2[mci];
                    localSuperActivity += nearestMiniColumnInfo.Temp_Activity;
                    //sumK += nearestMiniColumnInfo.Item1;
                }

                superActivity += NearestMiniColumnInfos[r].Item1 * localSuperActivity / NearestMiniColumnInfos[r].Item2.Count;
            }

            return superActivity;            
        }

        public void CalculateHash(float[] hash)
        {
            foreach (var bi in Enumerable.Range(0, hash.Length))
            {
                hash[bi] = 0.0f;
            }

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
        public MiniColumn? ActivityMax_MiniColumn = null;
        public float MaxSuperActivity = float.MinValue;
        public MiniColumn? SuperActivityMax_MiniColumn = null;
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

        float MiniColumnMinimumActivity { get; }

        /// <summary>
        ///     Верхний предел количества воспоминаний (для кэширования)
        /// </summary>
        int MemoriesMaxCount { get; }
    }
}
