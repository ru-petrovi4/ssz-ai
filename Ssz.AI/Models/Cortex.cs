using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Ssz.AI.Models
{
    public class Cortex : IOwnedDataSerializable
    {
        /// <summary>
        ///     
        /// </summary>
        /// <param name="constants"></param>        
        public Cortex(
            ICortexConstants constants,
            Retina retina)
        {
            Constants = constants;

            Random random = new();

            DetectorsVisibleRadius = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.DetectorDelta * constants.DetectorDelta / Math.PI);

            MiniColumns = new DenseTensor<MiniColumn>(constants.CortexWidth, constants.CortexHeight);
            double minCenterX = DetectorsVisibleRadius;
            double maxCenterX = constants.ImageWidth - DetectorsVisibleRadius;
            double deltaCenterX = (maxCenterX - minCenterX) / (constants.CortexWidth - 1);
            double minCenterY = DetectorsVisibleRadius;
            double maxCenterY = constants.ImageHeight - DetectorsVisibleRadius;
            double deltaCenterY = (maxCenterY - minCenterY) / (constants.CortexHeight - 1);                        

            foreach (int mcy in Enumerable.Range(0, MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, MiniColumns.Dimensions[0]))
                {
                    double centerX = minCenterX + mcx * deltaCenterX;
                    double centerY = minCenterY + mcy * deltaCenterY;

                    List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

                    for (int dy = (int)((centerY - DetectorsVisibleRadius) / constants.DetectorDelta); dy < (int)((centerY + DetectorsVisibleRadius) / constants.DetectorDelta) && dy < retina.Detectors.Dimensions[1]; dy += 1)
                        for (int dx = (int)((centerX - DetectorsVisibleRadius) / constants.DetectorDelta); dx < (int)((centerX + DetectorsVisibleRadius) / constants.DetectorDelta) && dx < retina.Detectors.Dimensions[0]; dx += 1)
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
                }            

            // Находим ближайшие миниколонки для каждой миниколонки
            Parallel.For(
                fromInclusive: 0,
                toExclusive: MiniColumns.Data.Length,
                mci =>
                {
                    MiniColumn mc = MiniColumns.Data[mci];

                    mc.NearestMiniColumnInfos.Add(((1.0f, 1.0f), new List<MiniColumn>(8)));

                    for (int mcy = mc.MCY - 1; mcy < mc.MCY + 1; mcy += 1)
                        for (int mcx = mc.MCX - 1; mcx < mc.MCX + 1; mcx += 1)
                        {
                            if (mcx < 0 ||
                                    mcx >= constants.CortexWidth ||
                                    mcy < 0 ||
                                    mcy >= constants.CortexHeight ||
                                    (mcx == mc.MCX && mcy == mc.MCY))
                                continue;

                            MiniColumn nearestMc = MiniColumns[mcx, mcy];

                            mc.NearestMiniColumnInfos[0].Item2.Add(nearestMc);
                        }
                });            
        }

        /// <summary>
        ///     In big picture coordinates
        /// </summary>
        public double DetectorsVisibleRadius { get; }

        public ICortexConstants Constants { get; }

        public DenseTensor<MiniColumn> MiniColumns { get; }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            if (context as string == "autoencoders")
            {
                using (writer.EnterBlock(1))
                {
                    writer.Write(MiniColumns.Data.Length);
                    for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
                    {
                        MiniColumn miniColumn = MiniColumns.Data[mci];
                        writer.WriteOwnedDataSerializableAndRecreatable(miniColumn.Autoencoder, null);
                    }
                }
            }            
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            if (context as string == "autoencoders")
            {
                using (Block block = reader.EnterBlock())
                {
                    switch (block.Version)
                    {
                        case 1:
                            int miniColumnsDataLength = reader.ReadInt32();
                            for (int mci = 0; mci < miniColumnsDataLength; mci += 1)
                            {
                                MiniColumn miniColumn = MiniColumns.Data[mci];
                                miniColumn.Autoencoder = reader.ReadOwnedDataSerializableAndRecreatable<Autoencoder>(null);
                            }
                            break;
                    }
                }
            }
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

            public Autoencoder? Autoencoder;

            /// <summary>
            ///     Текущий хэш активных детекторов при подаче примера.
            /// </summary>
            public readonly float[] Temp_Hash;            

            public int Temp_IterationsCount;

            public float Temp_CosineSimilarity;

            public float Temp_ControlCosineSimilarity;            

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

        public interface ICortexConstants : IRetinaConstants
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
            ///     Длина хэш-вектора
            /// </summary>
            int HashLength { get; }            

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            int MinBitsInHashForMemory { get; }            

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            int MemoriesMaxCount { get; }
        }
    }
}
