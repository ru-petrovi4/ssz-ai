using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models
{
    public class PreCortex : ISerializableModelObject
    {
        /// <summary>
        ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
        /// </summary>
        /// <param name="constants"></param>        
        public PreCortex(
            Model9.ModelConstants constants, 
            Eye leftEye,
            Eye rightEye)
        {
            Constants = constants;

            DetectorCorrelations = new DetectorCorrelation[rightEye.Retina.Detectors.Data.Length];

            foreach (var di in Enumerable.Range(0, rightEye.Retina.Detectors.Data.Length))
            {
                Detector detetctor = rightEye.Retina.Detectors.Data[di];
                DetectorCorrelation detectorCorrelation = new();

                detectorCorrelation.RangeLeftUpperX = detetctor.X - Constants.DependantDetectorsRangeWidthCount / 2;
                if (detectorCorrelation.RangeLeftUpperX < 0)
                    detectorCorrelation.RangeLeftUpperX = 0;

                detectorCorrelation.RangeLeftUpperY = detetctor.Y - Constants.DependantDetectorsRangeHeightCount / 2;
                if (detectorCorrelation.RangeLeftUpperY < 0)
                    detectorCorrelation.RangeLeftUpperY = 0;

                detectorCorrelation.RangeRightBottomX = detetctor.X + Constants.DependantDetectorsRangeWidthCount / 2;
                if (detectorCorrelation.RangeRightBottomX > leftEye.Retina.Detectors.Dimensions[0])
                    detectorCorrelation.RangeRightBottomX = leftEye.Retina.Detectors.Dimensions[0];

                detectorCorrelation.RangeRightBottomY = detetctor.Y + Constants.DependantDetectorsRangeHeightCount / 2;
                if (detectorCorrelation.RangeRightBottomY > leftEye.Retina.Detectors.Dimensions[1])
                    detectorCorrelation.RangeRightBottomY = leftEye.Retina.Detectors.Dimensions[1];

                detectorCorrelation.CorrelationMatrix = new MatrixFloat(
                    detectorCorrelation.RangeRightBottomX - detectorCorrelation.RangeLeftUpperX,
                    detectorCorrelation.RangeRightBottomY - detectorCorrelation.RangeLeftUpperY);

                DetectorCorrelations[di] = detectorCorrelation;
            }
        }
       
        public Model9.ModelConstants Constants { get; }

        public DetectorCorrelation[] DetectorCorrelations { get; }

        public void GenerateOwnedData()
        {           
        }

        public void Prepare()
        {
            
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        
                        break;
                }
            }
        }

        public void Calculate(
            StereoInput stereoInput, 
            Eye leftEye,
            Eye rightEye)
        {
            StereoInputItem[] stereoInputItems = stereoInput.StereoInputItems;

            foreach (var i in Enumerable.Range(0, 5000))
            {
                var stereoInputItem = stereoInputItems[i];

                var leftEye_GradientMatrix = stereoInputItem.LeftEye_GradientMatrix;
                var leftEye_Detectors = leftEye.Retina.Detectors;
                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: leftEye_Detectors.Data.Length,
                    di =>
                    {
                        var d = leftEye_Detectors.Data[di];
                        d.CalculateIsActivated(leftEye_GradientMatrix, Constants);
                    });
                
                var rightEye_GradientMatrix = stereoInputItem.RightEye_GradientMatrix;
                var rightEye_Detectors = rightEye.Retina.Detectors;
                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: rightEye_Detectors.Data.Length,
                    di =>
                    {
                        var d = rightEye_Detectors.Data[di];
                        d.CalculateIsActivated(rightEye_GradientMatrix, Constants);
                    });

                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: rightEye_Detectors.Data.Length,
                    di =>
                    {
                        var rightEye_Detector = rightEye_Detectors.Data[di];
                        var detectorCorrelation = DetectorCorrelations[di];
                        for (int dy = detectorCorrelation.RangeLeftUpperY; dy < detectorCorrelation.RangeRightBottomY; dy += 1)
                        {
                            for (int dx = detectorCorrelation.RangeLeftUpperX; dx < detectorCorrelation.RangeRightBottomX; dx += 1)
                            {
                                var leftEye_Detector_IsActivated = leftEye_Detectors[dx, dy].Temp_IsActivated;
                                if (leftEye_Detector_IsActivated && rightEye_Detector.Temp_IsActivated)
                                {
                                    detectorCorrelation.CorrelationMatrix[dx - detectorCorrelation.RangeLeftUpperX, dy - detectorCorrelation.RangeLeftUpperY] += 1.0f;
                                }
                            }
                        }   
                    });
            }
        }

        public struct DetectorCorrelation
        {
            public int RangeLeftUpperX;
            public int RangeLeftUpperY;
            public int RangeRightBottomX;
            public int RangeRightBottomY;
            public MatrixFloat CorrelationMatrix;
        }
    }    
}


//public class PreCortex : IOwnedDataSerializable
//{
//    /// <summary>
//    ///     
//    /// </summary>
//    /// <param name="constants"></param>        
//    public PreCortex(
//        IPreCortexConstants constants,
//        Retina retina)
//    {
//        Constants = constants;

//        Random random = new();

//        DetectorsVisibleRadius = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.DetectorDelta * constants.DetectorDelta / Math.PI);

//        MiniColumns = new DenseTensor<MiniColumn>(constants.PreCortexWidth, constants.PreCortexHeight);
//        double minCenterX = DetectorsVisibleRadius;
//        double maxCenterX = constants.ImageWidth - DetectorsVisibleRadius;
//        double deltaCenterX = (maxCenterX - minCenterX) / (constants.PreCortexWidth - 1);
//        double minCenterY = DetectorsVisibleRadius;
//        double maxCenterY = constants.ImageHeight - DetectorsVisibleRadius;
//        double deltaCenterY = (maxCenterY - minCenterY) / (constants.PreCortexHeight - 1);                        

//        foreach (int mcy in Enumerable.Range(0, MiniColumns.Dimensions[1]))
//            foreach (int mcx in Enumerable.Range(0, MiniColumns.Dimensions[0]))
//            {
//                double centerX = minCenterX + mcx * deltaCenterX;
//                double centerY = minCenterY + mcy * deltaCenterY;

//                List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

//                for (int dy = (int)((centerY - DetectorsVisibleRadius) / constants.DetectorDelta); dy < (int)((centerY + DetectorsVisibleRadius) / constants.DetectorDelta) && dy < retina.Detectors.Dimensions[1]; dy += 1)
//                    for (int dx = (int)((centerX - DetectorsVisibleRadius) / constants.DetectorDelta); dx < (int)((centerX + DetectorsVisibleRadius) / constants.DetectorDelta) && dx < retina.Detectors.Dimensions[0]; dx += 1)
//                    {
//                        Detector detector = retina.Detectors[dx, dy];
//                        double r = Math.Sqrt((detector.CenterX - centerX) * (detector.CenterX - centerX) + (detector.CenterY - centerY) * (detector.CenterY - centerY));
//                        if (r < DetectorsVisibleRadius)
//                            miniColumnDetectors.Add(detector);
//                    }

//                MiniColumn miniColumn = new MiniColumn(
//                    constants,
//                    mcx,
//                    mcy,
//                    miniColumnDetectors,
//                    centerX,
//                    centerY,
//                    random);

//                MiniColumns[mcx, mcy] = miniColumn;
//            }            

//        // Находим ближайшие миниколонки для каждой миниколонки
//        Parallel.For(
//            fromInclusive: 0,
//            toExclusive: MiniColumns.Data.Length,
//            mci =>
//            {
//                MiniColumn mc = MiniColumns.Data[mci];

//                mc.NearestMiniColumnInfos.Add(((1.0f, 1.0f), new List<MiniColumn>(8)));

//                for (int mcy = mc.MCY - 1; mcy < mc.MCY + 1; mcy += 1)
//                    for (int mcx = mc.MCX - 1; mcx < mc.MCX + 1; mcx += 1)
//                    {
//                        if (mcx < 0 ||
//                                mcx >= constants.PreCortexWidth ||
//                                mcy < 0 ||
//                                mcy >= constants.PreCortexHeight ||
//                                (mcx == mc.MCX && mcy == mc.MCY))
//                            continue;

//                        MiniColumn nearestMc = MiniColumns[mcx, mcy];

//                        mc.NearestMiniColumnInfos[0].Item2.Add(nearestMc);
//                    }
//            });            
//    }

//    /// <summary>
//    ///     In big picture coordinates
//    /// </summary>
//    public double DetectorsVisibleRadius { get; }

//    public IPreCortexConstants Constants { get; }

//    public DenseTensor<MiniColumn> MiniColumns { get; }

//    public void SerializeOwnedData(SerializationWriter writer, object? context)
//    {
//        if (context as string == "autoencoders")
//        {
//            using (writer.EnterBlock(1))
//            {
//                writer.Write(MiniColumns.Data.Length);
//                for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
//                {
//                    MiniColumn miniColumn = MiniColumns.Data[mci];
//                    writer.WriteOwnedDataSerializableAndRecreatable(miniColumn.Autoencoder, null);
//                }
//            }
//        }            
//    }

//    public void DeserializeOwnedData(SerializationReader reader, object? context)
//    {
//        if (context as string == "autoencoders")
//        {
//            using (Block block = reader.EnterBlock())
//            {
//                switch (block.Version)
//                {
//                    case 1:
//                        int miniColumnsDataLength = reader.ReadInt32();
//                        for (int mci = 0; mci < miniColumnsDataLength; mci += 1)
//                        {
//                            MiniColumn miniColumn = MiniColumns.Data[mci];
//                            miniColumn.Autoencoder = reader.ReadOwnedDataSerializableAndRecreatable<Autoencoder>(null);
//                        }
//                        break;
//                }
//            }
//        }
//    }

//    public class MiniColumn
//    {
//        public MiniColumn(IPreCortexConstants constants, int mcx, int mcy, List<Detector> detectors, double centerX, double centerY, Random random)
//        {
//            Constants = constants;
//            Detectors = detectors;
//            MCX = mcx;
//            MCY = mcy;
//            CenterX = centerX;
//            CenterY = centerY;
//            Temp_Hash = new float[constants.HashLength];

//            Memories = new(constants.MemoriesMaxCount); // { new Memory { Hash = hash0 } };
//            Temp_Memories = new(constants.MemoriesMaxCount);

//            NearestMiniColumnInfos = new List<((float, float), List<MiniColumn>)>();
//        }

//        public readonly IPreCortexConstants Constants;

//        public readonly List<Detector> Detectors;

//        /// <summary>
//        ///     Индекс миниколонки в матрице по оси X (горизонтально вправо)
//        /// </summary>
//        public readonly int MCX;

//        /// <summary>
//        ///     Индекс миниколонки в матрице по оси Y (вертикально вниз)
//        /// </summary>
//        public readonly int MCY;

//        /// <summary>
//        ///     (Величина, обратно пропорциональная расстоянию; List MiniColumn)
//        /// </summary>
//        public readonly List<((float, float), List<MiniColumn>)> NearestMiniColumnInfos;

//        /// <summary>
//        ///     [0..MNISTImageWidth]
//        /// </summary>
//        public readonly double CenterX;

//        /// <summary>
//        ///     [0..MNISTImageHeight]
//        /// </summary>
//        public readonly double CenterY;

//        /// <summary>
//        ///     Сохраненные хэш-коды
//        /// </summary>
//        public List<Memory> Memories;            

//        /// <summary>
//        ///     Временный список для сохраненных хэш-кодов
//        /// </summary>
//        public List<Memory> Temp_Memories;

//        public Autoencoder? Autoencoder;

//        /// <summary>
//        ///     Текущий хэш активных детекторов при подаче примера.
//        /// </summary>
//        public readonly float[] Temp_Hash;            

//        public int Temp_IterationsCount;

//        public float Temp_CosineSimilarity;

//        public float Temp_ControlCosineSimilarity;            

//        public void CalculateHash(float[] hash)
//        {
//            Array.Clear(hash);

//            foreach (var detector in Detectors)
//            {
//                if (detector.Temp_IsActivated)
//                    hash[detector.BitIndexInHash] = 1.0f;
//            }
//        }            
//    }

//    public struct Memory
//    {
//        public float[] Hash;
//        public bool IsDeleted;
//    }

//    public interface IPreCortexConstants : IRetinaConstants
//    {
//        /// <summary>
//        ///     Количество миниколонок в зоне коры по оси X
//        /// </summary>
//        int PreCortexWidth { get; }

//        /// <summary>
//        ///     Количество миниколонок в зоне коры по оси Y
//        /// </summary>
//        int PreCortexHeight { get; }

//        /// <summary>
//        ///     Ширина основного изображения
//        /// </summary>
//        int ImageWidth { get; }

//        /// <summary>
//        ///     Высота основного изображения
//        /// </summary>
//        int ImageHeight { get; }

//        /// <summary>
//        ///     Количество детекторов, видимых одной миниколонкой
//        /// </summary>
//        int MiniColumnVisibleDetectorsCount { get; }               

//        /// <summary>
//        ///     Длина хэш-вектора
//        /// </summary>
//        int HashLength { get; }            

//        /// <summary>
//        ///     Минимальное число бит в хэше, что бы быть сохраненным в память
//        /// </summary>
//        int MinBitsInHashForMemory { get; }            

//        /// <summary>
//        ///     Верхний предел количества воспоминаний (для кэширования)
//        /// </summary>
//        int MemoriesMaxCount { get; }
//    }
//}