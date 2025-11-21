using Ssz.AI.Helpers;
using Ssz.Utils;
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
    public class Cortex : ISerializableModelObject
    {
        /// <summary>
        ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
        /// </summary>
        /// <param name="constants"></param>        
        public Cortex(
            Model11.ModelConstants constants, 
            Eye leftEye,
            Eye rightEye)
        {
            Constants = constants;

            DetectorCorrelations = new DetectorCorrelation[rightEye.Retina.Detectors.Data.Length];

            //foreach (var di in Enumerable.Range(0, rightEye.Retina.Detectors.Data.Length))
            //{
            //    Detector detetctor = rightEye.Retina.Detectors.Data[di];
            //    DetectorCorrelation detectorCorrelation = new();

            //    detectorCorrelation.RangeLeftUpperX = detetctor.DetectorX - Constants.DependantDetectorsRangeWidthCount / 2;
            //    if (detectorCorrelation.RangeLeftUpperX < 0)
            //        detectorCorrelation.RangeLeftUpperX = 0;

            //    detectorCorrelation.RangeLeftUpperY = detetctor.DetectorY - Constants.DependantDetectorsRangeHeightCount / 2;
            //    if (detectorCorrelation.RangeLeftUpperY < 0)
            //        detectorCorrelation.RangeLeftUpperY = 0;

            //    detectorCorrelation.RangeRightBottomX = detetctor.DetectorX + Constants.DependantDetectorsRangeWidthCount / 2;
            //    if (detectorCorrelation.RangeRightBottomX > leftEye.Retina.Detectors.Dimensions[0])
            //        detectorCorrelation.RangeRightBottomX = leftEye.Retina.Detectors.Dimensions[0];

            //    detectorCorrelation.RangeRightBottomY = detetctor.DetectorY + Constants.DependantDetectorsRangeHeightCount / 2;
            //    if (detectorCorrelation.RangeRightBottomY > leftEye.Retina.Detectors.Dimensions[1])
            //        detectorCorrelation.RangeRightBottomY = leftEye.Retina.Detectors.Dimensions[1];

            //    detectorCorrelation.CorrelationMatrix = new MatrixFloat(
            //        detectorCorrelation.RangeRightBottomX - detectorCorrelation.RangeLeftUpperX,
            //        detectorCorrelation.RangeRightBottomY - detectorCorrelation.RangeLeftUpperY);

            //    DetectorCorrelations[di] = detectorCorrelation;
            //}

            DetectorsVisibleRadiusPixels = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.RetinaDetectorsDeltaPixels * constants.RetinaDetectorsDeltaPixels / Math.PI);

            MiniColumns = new DenseMatrix<MiniColumn>(constants.CortexWidth_MiniColumns, constants.CortexHeight_MiniColumns);
            double minCenterXPixels = DetectorsVisibleRadiusPixels;
            double maxCenterXPixels = constants.RetinaImagePixelSize.Width - DetectorsVisibleRadiusPixels;
            double deltaCenterXPixels = (maxCenterXPixels - minCenterXPixels) / (constants.CortexWidth_MiniColumns - 1);
            double minCenterYPixels = DetectorsVisibleRadiusPixels;
            double maxCenterYPixels = constants.RetinaImagePixelSize.Height - DetectorsVisibleRadiusPixels;
            double deltaCenterYPixels = (maxCenterYPixels - minCenterYPixels) / (constants.CortexHeight_MiniColumns - 1);

            if (constants.CalculationsSubAreaRadius_MiniColumns is not null)
                CalculationSubAreaRadius_MiniColumns_Value = ((IConstants)constants).CalculationsSubArea_MiniColumns_Count;
            else
                CalculationSubAreaRadius_MiniColumns_Value = Math.Sqrt(MiniColumns.Data.Length / Math.PI) * 2.0f;

            // Создаем только миниколонки для подобласти            
            foreach (int mcy in Enumerable.Range(0, constants.CortexHeight_MiniColumns))
                foreach (int mcx in Enumerable.Range(0, constants.CortexWidth_MiniColumns))
                {
                    double miniColumnR = Math.Sqrt((mcx - constants.CalculationsSubAreaCenter_Cx) * (mcx - constants.CalculationsSubAreaCenter_Cx) + (mcy - constants.CalculationsSubAreaCenter_Cy) * (mcy - constants.CalculationsSubAreaCenter_Cy));
                    if (miniColumnR < CalculationSubAreaRadius_MiniColumns_Value)
                    {
                        double centerXPixels = minCenterXPixels + mcx * deltaCenterXPixels;
                        double centerYPixels = minCenterYPixels + mcy * deltaCenterYPixels;

                        List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

                        for (int detectorY = (int)((centerYPixels - DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels); detectorY < (int)((centerYPixels + DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels) && detectorY < leftEye.Retina.Detectors.Dimensions[1]; detectorY += 1)
                            for (int detectorX = (int)((centerXPixels - DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels); detectorX < (int)((centerXPixels + DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels) && detectorX < leftEye.Retina.Detectors.Dimensions[0]; detectorX += 1)
                            {
                                Detector detector = leftEye.Retina.Detectors[detectorX, detectorY];
                                double rPixels = Math.Sqrt((detector.CenterXPixels - centerXPixels) * (detector.CenterXPixels - centerXPixels) + (detector.CenterYPixels - centerYPixels) * (detector.CenterYPixels - centerYPixels));
                                if (rPixels < DetectorsVisibleRadiusPixels)
                                    miniColumnDetectors.Add(detector);
                            }

                        for (int detectorY = (int)((centerYPixels - DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels); detectorY < (int)((centerYPixels + DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels) && detectorY < rightEye.Retina.Detectors.Dimensions[1]; detectorY += 1)
                            for (int detectorX = (int)((centerXPixels - DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels); detectorX < (int)((centerXPixels + DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels) && detectorX < rightEye.Retina.Detectors.Dimensions[0]; detectorX += 1)
                            {
                                Detector detector = rightEye.Retina.Detectors[detectorX, detectorY];
                                double rPixels = Math.Sqrt((detector.CenterXPixels - centerXPixels) * (detector.CenterXPixels - centerXPixels) + (detector.CenterYPixels - centerYPixels) * (detector.CenterYPixels - centerYPixels));
                                if (rPixels < DetectorsVisibleRadiusPixels)
                                    miniColumnDetectors.Add(detector);
                            }

                        MiniColumn miniColumn = new MiniColumn(
                            constants,
                            mcx,
                            mcy,
                            miniColumnDetectors,
                            centerXPixels,
                            centerYPixels);

                        MiniColumns[mcx, mcy] = miniColumn;

                        if (miniColumnR < 0.000001)
                            CenterMiniColumn = miniColumn;
                    }
                }

            HashSet<Detector> subAreaOrAll_DetectorsHashSet = new(2 * leftEye.Retina.Detectors.Dimensions[0] * leftEye.Retina.Detectors.Dimensions[1]);
            List<MiniColumn> subAreaOrAll_MiniColumns = new(constants.CalculationsSubAreaRadius_MiniColumns is not null ? ((IConstants)constants).CalculationsSubArea_MiniColumns_Count : (MiniColumns.Dimensions[0] * MiniColumns.Dimensions[1]));

            foreach (int mcy in Enumerable.Range(0, MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, MiniColumns.Dimensions[0]))
                {
                    var mc = MiniColumns[mcx, mcy];
                    if (mc is not null)
                    {
                        subAreaOrAll_MiniColumns.Add(mc);
                        foreach (var d in mc.Detectors)
                        {
                            subAreaOrAll_DetectorsHashSet.Add(d);
                        }
                    }
                }
            SubAreaOrAll_MiniColumns = subAreaOrAll_MiniColumns.ToArray();
            SubAreaOrAll_Detectors = subAreaOrAll_DetectorsHashSet.ToArray();
            
            // Находим ближайшие миниколонки для каждой миниколонки
            Parallel.For(
                fromInclusive: 0,
                toExclusive: SubAreaOrAll_MiniColumns.Length,
                mci =>
                {
                    MiniColumn mc = SubAreaOrAll_MiniColumns[mci];
                    
                    mc.K0 = (constants.PositiveK[0], constants.NegativeK[0]);

                    for (int mcy = mc.MCY - (int)constants.SuperActivityRadius_MiniColumns - 1; mcy <= mc.MCY + (int)constants.SuperActivityRadius_MiniColumns + 1; mcy += 1)
                        for (int mcx = mc.MCX - (int)constants.SuperActivityRadius_MiniColumns - 1; mcx <= mc.MCX + (int)constants.SuperActivityRadius_MiniColumns + 1; mcx += 1)
                        {
                            if (mcx < 0 ||
                                    mcx >= constants.CortexWidth_MiniColumns ||
                                    mcy < 0 ||
                                    mcy >= constants.CortexHeight_MiniColumns ||
                                    (mcx == mc.MCX && mcy == mc.MCY))
                                continue;

                            MiniColumn nearestMc = MiniColumns[mcx, mcy];
                            if (nearestMc is null)
                                continue;
                            float r = MathF.Sqrt((mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY));
                            if (r < constants.SuperActivityRadius_MiniColumns + 0.00001f)
                            {                                
                                float k0 = MathHelper.GetInterpolatedValue(constants.PositiveK, r);
                                float k1 = MathHelper.GetInterpolatedValue(constants.NegativeK, r);
                                mc.K_ForNearestMiniColumns.Add((k0, k1, nearestMc));
                            }
                        }

                    for (int mcy = mc.MCY - (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns - 1; mcy <= mc.MCY + (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns + 1; mcy += 1)
                        for (int mcx = mc.MCX - (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns - 1; mcx <= mc.MCX + (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns + 1; mcx += 1)
                        {
                            if (mcx < 0 ||
                                    mcx >= constants.CortexWidth_MiniColumns ||
                                    mcy < 0 ||
                                    mcy >= constants.CortexHeight_MiniColumns)
                                continue;

                            MiniColumn nearestMc = MiniColumns[mcx, mcy];
                            if (nearestMc is null)
                                continue;
                            float r = MathF.Sqrt((mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY));
                            if (r < constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns + 0.00001f)
                                mc.NearestMiniColumnsAndSelf_ForMemorySaving.Add(nearestMc);
                        }
                });

            VisualizationTableItems = new(1000);

            //InputAutoencoder = new Autoencoder();
        }
       
        public Model11.ModelConstants Constants { get; }

        public DetectorCorrelation[] DetectorCorrelations { get; }

        public DenseMatrix<MiniColumn> MiniColumns;

        public double DetectorsVisibleRadiusPixels { get; }

        public MiniColumn? CenterMiniColumn { get; private set; }

        /// <summary>
        ///     Sub Area or All MiniColumns
        /// </summary>
        public MiniColumn[] SubAreaOrAll_MiniColumns { get; } = null!;

        /// <summary>
        ///     Sub Area or All Detectors
        /// </summary>
        public Detector[] SubAreaOrAll_Detectors { get; } = null!;

        public double CalculationSubAreaRadius_MiniColumns_Value;

        public Autoencoder InputAutoencoder { get; } = null!;

        public List<VisualizationTableItem> VisualizationTableItems { get; }

        public MiniColumn? Temp_SuperActivityMax_MiniColumn
        {
            get;
            set;
        }

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
            //StereoInputItem[] stereoInputItems = stereoInput.StereoInputItems;

            //foreach (var i in Enumerable.Range(0, 5000))
            //{
            //    var stereoInputItem = stereoInputItems[i];

            //    var leftEye_GradientMatrix = stereoInputItem.LeftEye_GradientMatrix;
            //    var leftEye_Detectors = leftEye.Retina.Detectors;
            //    Parallel.For(
            //        fromInclusive: 0,
            //        toExclusive: leftEye_Detectors.Data.Length,
            //        di =>
            //        {
            //            var d = leftEye_Detectors.Data[di];
            //            d.CalculateIsActivated(leftEye.Retina, leftEye_GradientMatrix, Constants);
            //        });
                
            //    var rightEye_GradientMatrix = stereoInputItem.RightEye_GradientMatrix;
            //    var rightEye_Detectors = rightEye.Retina.Detectors;
            //    Parallel.For(
            //        fromInclusive: 0,
            //        toExclusive: rightEye_Detectors.Data.Length,
            //        di =>
            //        {
            //            var d = rightEye_Detectors.Data[di];
            //            d.CalculateIsActivated(rightEye.Retina, rightEye_GradientMatrix, Constants);
            //        });

            //    Parallel.For(
            //        fromInclusive: 0,
            //        toExclusive: rightEye_Detectors.Data.Length,
            //        di =>
            //        {
            //            var rightEye_Detector = rightEye_Detectors.Data[di];
            //            var detectorCorrelation = DetectorCorrelations[di];
            //            for (int dy = detectorCorrelation.RangeLeftUpperY; dy < detectorCorrelation.RangeRightBottomY; dy += 1)
            //            {
            //                for (int dx = detectorCorrelation.RangeLeftUpperX; dx < detectorCorrelation.RangeRightBottomX; dx += 1)
            //                {
            //                    var leftEye_Detector_IsActivated = leftEye_Detectors[dx, dy].Temp_IsActivated;
            //                    if (leftEye_Detector_IsActivated && rightEye_Detector.Temp_IsActivated)
            //                    {
            //                        detectorCorrelation.CorrelationMatrix[dx - detectorCorrelation.RangeLeftUpperX, dy - detectorCorrelation.RangeLeftUpperY] += 1.0f;
            //                    }
            //                }
            //            }   
            //        });
            //}
        }

        /// <summary>
        ///     
        /// </summary>
        /// <param name="action"></param>
        public async Task DoSafeCalculationsAsync(Action<MiniColumn> action, int radius_MiniColumns)
        {
            //int processorCount = Environment.ProcessorCount;
            int rangeHeight = radius_MiniColumns * 2;

            int fullRangesCount = Constants.CortexHeight_MiniColumns / rangeHeight;
            int fullRangesFullPairsCount = fullRangesCount / 2;

            List<Task> tasks = new();
            for (int rangePairIndex = 0; rangePairIndex < fullRangesFullPairsCount + 1; rangePairIndex += 1)
            {
                tasks.Add(Task.Run(() =>
                {
                    int mcyFromInclusive = (rangePairIndex * 2 + 0) * rangeHeight;
                    int mcyToExclusive = Math.Min((rangePairIndex * 2 + 1) * rangeHeight, Constants.CortexHeight_MiniColumns);
                    for (int mcy = mcyFromInclusive; mcy < mcyToExclusive; mcy += 1)
                        for (int mcx = 0; mcx < Constants.CortexWidth_MiniColumns; mcx += 1)
                        {
                            MiniColumn? miniColumn = MiniColumns[mcx, mcy];
                            if (miniColumn is not null)
                                action(miniColumn);
                        }
                }));
            }
            await Task.WhenAll(tasks);

            tasks.Clear();
            for (int rangePairIndex = 0; rangePairIndex < fullRangesFullPairsCount + 1; rangePairIndex += 1)
            {
                tasks.Add(Task.Run(() =>
                {
                    int mcyFromInclusive = (rangePairIndex * 2 + 1) * rangeHeight;
                    int mcyToExclusive = Math.Min((rangePairIndex * 2 + 2) * rangeHeight, Constants.CortexHeight_MiniColumns);
                    for (int mcy = mcyFromInclusive; mcy < mcyToExclusive; mcy += 1)
                        for (int mcx = 0; mcx < Constants.CortexWidth_MiniColumns; mcx += 1)
                        {
                            MiniColumn? miniColumn = MiniColumns[mcx, mcy];
                            if (miniColumn is not null)
                                action(miniColumn);
                        }
                }));
            }
            await Task.WhenAll(tasks);
        }

        public struct DetectorCorrelation
        {
            public int RangeLeftUpperX;
            public int RangeLeftUpperY;
            public int RangeRightBottomX;
            public int RangeRightBottomY;
            public MatrixFloat_ColumnMajor CorrelationMatrix;
        }

        public class MiniColumn : IMiniColumn, IMiniColumnActivity, ISerializableModelObject
        {
            public MiniColumn(IConstants constants, int mcx, int mcy, List<Detector> detectors, double centerXPixels, double centerYPixels)
            {
                Constants = constants;
                Detectors = detectors;
                MCX = mcx;
                MCY = mcy;
                CenterXPixels = centerXPixels;
                CenterYPixels = centerYPixels;
                Temp_Hash = new float[constants.HashLength];
                Memories = new(constants.MemoriesMaxCount);
                Temp_Memories = new(constants.MemoriesMaxCount);
                Temp_ShortHashConverted = new float[constants.ShortHashLength];

                K_ForNearestMiniColumns = new FastList<(float, float, IMiniColumnActivity)>((int)(Math.PI * constants.SuperActivityRadius_MiniColumns * constants.SuperActivityRadius_MiniColumns) + 10);
                NearestMiniColumnsAndSelf_ForMemorySaving = new List<MiniColumn>((int)(Math.PI * constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns * 
                    constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns) + 10);
            }

            public readonly IConstants Constants;

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
            ///     K для расчета суперактивности.
            ///     (K для позитива, K для негатива, MiniColumn)
            /// </summary>
            public readonly FastList<(float, float, IMiniColumnActivity)> K_ForNearestMiniColumns;

            /// <summary>
            ///     Minicolums in field of radius of <see cref="IConstants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns"/>
            /// </summary>
            public readonly List<MiniColumn> NearestMiniColumnsAndSelf_ForMemorySaving;

            /// <summary>
            ///     K0 для расчета суперактивности.
            /// </summary>
            public (float, float) K0;

            /// <summary>
            ///     [0..RetinaImagePixelSize.Width]
            /// </summary>
            public readonly double CenterXPixels;

            /// <summary>
            ///     [0..RetinaImagePixelSize.Height]
            /// </summary>
            public readonly double CenterYPixels;

            /// <summary>
            ///     Сохраненные хэш-коды
            /// </summary>
            public FastList<Memory?> Memories;

            /// <summary>
            ///     Временный список для сохраненных хэш-кодов
            /// </summary>
            public FastList<Memory?> Temp_Memories;

            /// <summary>
            ///     Последнее добавленное воспомининие            
            /// </summary>
            public Memory? Temp_Memory;

            public List<List<Memory>> Temp_MemoryClusters = new();

            public Autoencoder? Autoencoder;

            /// <summary>
            ///     Преобразование из индекса Temp_ShortHash в индекс в Temp_ShortHashConverted
            /// </summary>
            public int[]? ShortHashConversion;

            /// <summary>
            ///     Текущая активность миниколонки при подаче примера.
            ///     (Позитивная активность, Негативная активность, Количество воспоминаний)
            ///     Всегда не NaN.
            /// </summary>
            public (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Temp_Activity;

            /// <summary>
            ///     Текущая суммарная активность миниколонки с учетом активностей соседей при подаче примера
            ///     Может быть NaN
            /// </summary>
            public float Temp_SuperActivity;

            /// <summary>
            ///     Текущий хэш активных детекторов при подаче примера.
            /// </summary>
            public readonly float[] Temp_Hash;

            public bool Temp_MemoryCanBeStored;

            public float Temp_ShortHashConverted_SyncQuality;

            public float Temp_ShortHashConverted_SyncQualitySum;

            public int Temp_ShortHashConverted_SyncQualitySumCount;

            public float Temp_Hash_SyncQuality;

            public float Temp_Hash_SyncQualitySum;

            public int Temp_Hash_SyncQualitySumCount;

            public float Temp_ShortHash_AutoencoderSyncQuality;

            public float Temp_ShortHash_AutoencoderSyncQualitySum;

            public int Temp_ShortHash_AutoencoderSyncQualitySumCount;

            /// <summary>
            ///     Handle in ObjectMamager: SyncedMiniColumnsToProcess
            /// </summary>
            public UInt32 Temp_SyncedMiniColumnsToProcess_Handle;

            public MatrixFloat_ColumnMajor? Temp_ShortHashConversionMatrix;

            public int Temp_ShortHashConversionMatrix_TrainingCount;

            public readonly float[] Temp_ShortHashConverted;

            public bool Temp_IsShortHashMustBeCalculated;

            public void GetHash(float[] hash)
            {
                Array.Clear(hash);

                foreach (var detector in Detectors)
                {
                    if (detector.Temp_IsActivated)
                        hash[detector.BitIndexInHash] = 1.0f;
                }
            }

            public void AddMemory(Memory memory)
            {
                Temp_Memory = memory;
                Memories.Add(memory);
            }

            public void GetShortHashConverted(float[] shortHash, float[] shortHashConverted)
            {
                if (ShortHashConversion is not null)
                {
                    for (int i = 0; i < ShortHashConversion.Length; i++)
                    {
                        shortHashConverted[ShortHashConversion[i]] = shortHash[i];
                    }
                }
                else
                {
                    Array.Clear(shortHashConverted);
                    int onesCount = 0;
                    foreach (int i in Enumerable.Range(0, shortHash.Length))
                    {
                        if (shortHash[i] == 1.0f)
                        {
                            foreach (int j in Enumerable.Range(0, shortHashConverted.Length))
                            {
                                shortHashConverted[j] += Temp_ShortHashConversionMatrix![i, j];
                            }
                            onesCount += 1;
                        }
                    }
                    var indices = shortHashConverted
                        .Select((value, index) => (value, index))
                        .OrderByDescending(item => item.value)
                        .Take(onesCount)
                        .Select(item => item.index)
                        .ToHashSet();

                    for (int i = 0; i < shortHashConverted.Length; i++)
                    {
                        if (!indices.Contains(i))
                            shortHashConverted[i] = 0.0f;
                        else
                            shortHashConverted[i] = 1.0f;
                    }
                }
            }

            /// <summary>
            ///     Max, Min, Average
            /// </summary>
            /// <returns></returns>
            public (GradientInPoint, GradientInPoint, GradientInPoint) GetPictureAverageGradientInPoint()
            {
                GradientInPoint max = new GradientInPoint()
                {
                    GradX = Double.MinValue,
                    GradY = Double.MinValue,
                };
                GradientInPoint min = new GradientInPoint()
                {
                    GradX = Double.MaxValue,
                    GradY = Double.MaxValue,
                };
                double gradX = 0.0;
                double gradY = 0.0;
                int notNullCount = 0;
                foreach (var detector in Detectors)
                {
                    if (detector.Temp_GradientInPoint.GradX != 0 ||
                            detector.Temp_GradientInPoint.GradY != 0)
                    {
                        if (detector.Temp_GradientInPoint.GradX > max.GradX)
                            max.GradX = detector.Temp_GradientInPoint.GradX;
                        if (detector.Temp_GradientInPoint.GradY > max.GradY)
                            max.GradY = detector.Temp_GradientInPoint.GradY;
                        if (detector.Temp_GradientInPoint.GradX < min.GradX)
                            min.GradX = detector.Temp_GradientInPoint.GradX;
                        if (detector.Temp_GradientInPoint.GradY < min.GradY)
                            min.GradY = detector.Temp_GradientInPoint.GradY;
                        gradX += detector.Temp_GradientInPoint.GradX;
                        gradY += detector.Temp_GradientInPoint.GradY;
                        notNullCount += 1;
                    }
                }
                if (notNullCount > 0)
                {
                    gradX /= notNullCount;
                    gradY /= notNullCount;
                }
                double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                double angle = Math.Atan2(gradY, gradX); // Угол в радианах    
                return (max, min, new GradientInPoint
                {
                    GradX = gradX,
                    GradY = gradY,
                    Angle = angle,
                    Magnitude = magnitude
                });
            }

            public void SerializeOwnedData(SerializationWriter writer, object? context)
            {
                if (context as string == "autoencoders")
                {
                    using (writer.EnterBlock(1))
                    {
                        writer.WriteOwnedDataSerializableAndRecreatable(Autoencoder, null);
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
                                Autoencoder = reader.ReadOwnedDataSerializableAndRecreatable<Autoencoder>(null);
                                break;
                        }
                    }
                }
            }

            IFastList<ICortexMemory?> IMiniColumn.CortexMemories => Memories;

            IMiniColumn IMiniColumnActivity.MiniColumn => this;

            (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) IMiniColumnActivity.Activity => Temp_Activity;

            float IMiniColumnActivity.SuperActivity => Temp_SuperActivity;

            IFastList<(float, float, IMiniColumnActivity)> IMiniColumnActivity.K_ForNearestMiniColumns => K_ForNearestMiniColumns;
        }

        public class Memory : ICortexMemory
        {
            public float[] Hash = null!;

            public GradientInPoint PictureAverageGradientInPoint;

            /// <summary>
            ///     Input image index for this memory.
            /// </summary>
            public int PictureInputIndex;

            float[] ICortexMemory.DiscreteVector => Hash;
        }

        public class ActivitiyMaxInfo
        {
            public MiniColumn? Temp_WinnerMiniColumn;

            public float MaxActivity = float.MinValue;
            public readonly List<MiniColumn> ActivityMax_MiniColumns = new();

            public float MaxSuperActivity = float.MinValue;
            public readonly List<MiniColumn> SuperActivityMax_MiniColumns = new();
            public MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
            {
                if (SuperActivityMax_MiniColumns.Count == 0)
                    return null;
                if (SuperActivityMax_MiniColumns.Count == 1)
                    return SuperActivityMax_MiniColumns[0];
                var winnerIndex = random.Next(SuperActivityMax_MiniColumns.Count);
                Temp_WinnerMiniColumn = SuperActivityMax_MiniColumns[winnerIndex];
                return Temp_WinnerMiniColumn;
            }            
        }
    }    
}