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
            List<Detector> detectors)
        {
            Random random = new();

            double visibleRadius = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.DetectorArea / Math.PI);

            MiniColumns = new MiniColumn[constants.CortexWidth, constants.CortexHeight];
            double minCenterX = visibleRadius;
            double maxCenterX = constants.ImageWidth - visibleRadius;
            double deltaCenterX = (maxCenterX - minCenterX) / (constants.CortexWidth - 1);
            double minCenterY = visibleRadius;
            double maxCenterY = constants.ImageHeight - visibleRadius;
            double deltaCenterY = (maxCenterY - minCenterY) / (constants.CortexHeight - 1);


            double subAreaRadius;
            if (constants.SubAreaMiniColumnsCount is not null)
                subAreaRadius = Math.Sqrt(constants.SubAreaMiniColumnsCount.Value / Math.PI);
            else
                subAreaRadius = 0.0;

            Parallel.For(
                fromInclusive: 0,
                toExclusive: constants.CortexHeight,
                mcy =>
                {
                    for (int mcx = 0; mcx < constants.CortexWidth; mcx += 1)
                    {
                        if (subAreaRadius == 0.0 ||
                            Math.Sqrt(Math.Pow(mcx - constants.SubAreaCenter_Cx, 2) + Math.Pow(mcy - constants.SubAreaCenter_Cy, 2)) < subAreaRadius)
                        {
                            double centerX = minCenterX + mcx * deltaCenterX;
                            double centerY = minCenterY + mcy * deltaCenterY;

                            List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

                            foreach (var detector in detectors)
                            {
                                double r = Math.Sqrt(Math.Pow(detector.CenterX - centerX, 2) + Math.Pow(detector.CenterY - centerY, 2));
                                if (r < visibleRadius)
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
                        }
                    }
                });

            HashSet<Detector> subArea_DetectorsHashSet = new(detectors.Count);
            List<MiniColumn> subArea_MiniColums = new(constants.SubAreaMiniColumnsCount ?? (MiniColumns.GetLength(0) * MiniColumns.GetLength(1)));
            for (int mcy = 0; mcy < MiniColumns.GetLength(1); mcy += 1)
                for (int mcx = 0; mcx < MiniColumns.GetLength(0); mcx += 1)
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
                    for (int mcy = mc.MCY - constants.NearestMiniColumnsDelta; mcy < mc.MCY + constants.NearestMiniColumnsDelta; mcy += 1)
                        for (int mcx = mc.MCX - constants.NearestMiniColumnsDelta; mcx < mc.MCX + constants.NearestMiniColumnsDelta; mcx += 1)
                        {
                            if (mcx < 0 ||
                                    mcx == mc.MCX ||
                                    mcx >= constants.CortexWidth ||
                                    mcy < 0 ||
                                    mcy == mc.MCY ||
                                    mcy >= constants.CortexHeight)
                                continue;

                            MiniColumn nearestMc = MiniColumns[mcx, mcy];
                            if (nearestMc is null)
                                continue;
                            double r = Math.Sqrt(Math.Pow(mcx - mc.MCX, 2) + Math.Pow(mcy - mc.MCY, 2));
                            if (r < constants.NearestMiniColumnsDelta)
                            {
                                mc.NearestMiniColumns.Add((constants.NearestMiniColumnsK / r, nearestMc));
                            }                            
                        }
                });
        }

        public MiniColumn[,] MiniColumns { get; }

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
            Temp_Memories = new() { hash0 };
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
        ///     (Величина, обратно пропорциональная расстоянию; MiniColumn)
        /// </summary>
        public readonly List<(double, MiniColumn)> NearestMiniColumns = new();

        /// <summary>
        ///     [0..MNISTImageWidth]
        /// </summary>
        public readonly double CenterX;

        /// <summary>
        ///     [0..MNISTImageHeight]
        /// </summary>
        public readonly double CenterY;

        /// <summary>
        ///     Активность миниколонки при подаче примера
        /// </summary>
        public float Temp_Activity;

        /// <summary>
        ///     Суммарная активность миниколонки с учетом активностей соседей при подаче примера
        /// </summary>
        public float Temp_SuperActivity;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public readonly List<float[]> Temp_Memories;

        public readonly float[] Temp_Hash;

        public float GetActivity()
        {
            CalculateHash(Temp_Hash);

            if (TensorPrimitives.Sum(Temp_Hash) < Constants.MinBitsInHashForMemory)
                return 0.0f;

            float activity = 0.0f;

            foreach (var mi in Enumerable.Range(0, Temp_Memories.Count))
            {
                activity += TensorPrimitives.CosineSimilarity(Temp_Hash, Temp_Memories[mi]) - 0.66f;
            }

            return activity;
        }

        public float GetSuperActivity()
        {
            float superActivity = Temp_Activity;

            foreach (var mci in Enumerable.Range(0, NearestMiniColumns.Count))
            {
                var data = NearestMiniColumns[mci];
                superActivity += (float)(data.Item2.Temp_Activity * data.Item1);
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
        ///     Площадь одного детектрора   
        /// </summary>
        double DetectorArea { get; }    
        
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

        double NearestMiniColumnsK { get; }
    }
}
