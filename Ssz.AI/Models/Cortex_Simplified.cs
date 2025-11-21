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

namespace Ssz.AI.Models;

public class Cortex_Simplified : ISerializableModelObject
{
    /// <summary>
    ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
    /// </summary>
    /// <param name="constants"></param>        
    public Cortex_Simplified(
        IConstants constants,            
        Retina retina)
    {
        Constants = constants;

        //PositiveK = new float[Constants.MiniColumnsMaxDistance + 1];
        //NegativeK = new float[Constants.MiniColumnsMaxDistance + 1];

        DetectorsVisibleRadiusPixels = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.RetinaDetectorsDeltaPixels * constants.RetinaDetectorsDeltaPixels / Math.PI);

        MiniColumns = new DenseMatrix<MiniColumn>(constants.CortexWidth_MiniColumns, constants.CortexHeight_MiniColumns);
        double minCenterX = DetectorsVisibleRadiusPixels;
        double maxCenterX = MNISTHelper.MNISTImageWidthPixels - DetectorsVisibleRadiusPixels;
        double deltaCenterX = (maxCenterX - minCenterX) / (constants.CortexWidth_MiniColumns - 1);
        double minCenterY = DetectorsVisibleRadiusPixels;
        double maxCenterY = MNISTHelper.MNISTImageHeightPixels - DetectorsVisibleRadiusPixels;
        double deltaCenterY = (maxCenterY - minCenterY) / (constants.CortexHeight_MiniColumns - 1);
        
        if (constants.CalculationsSubAreaRadius_MiniColumns is not null)
            SubArea_MiniColumns_Radius = constants.CalculationsSubAreaRadius_MiniColumns.Value;
        else
            SubArea_MiniColumns_Radius = 0.0;            

        // Создаем только миниколонки для подобласти            
        foreach (int mcy in Enumerable.Range(0, constants.CortexHeight_MiniColumns))
            foreach (int mcx in Enumerable.Range(0, constants.CortexWidth_MiniColumns))
            {
                double miniColumnR = Math.Sqrt((mcx - constants.CalculationsSubAreaCenter_Cx) * (mcx - constants.CalculationsSubAreaCenter_Cx) + (mcy - constants.CalculationsSubAreaCenter_Cy) * (mcy - constants.CalculationsSubAreaCenter_Cy));
                if (SubArea_MiniColumns_Radius == 0.0 || miniColumnR < SubArea_MiniColumns_Radius)
                {
                    double centerX = minCenterX + mcx * deltaCenterX;
                    double centerY = minCenterY + mcy * deltaCenterY;

                    List<Detector> miniColumnDetectors = new(constants.MiniColumnVisibleDetectorsCount);

                    for (int dy = (int)((centerY - DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels); dy < (int)((centerY + DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels) && dy < retina.Detectors.Dimensions[1]; dy += 1)
                        for (int dx = (int)((centerX - DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels); dx < (int)((centerX + DetectorsVisibleRadiusPixels) / constants.RetinaDetectorsDeltaPixels) && dx < retina.Detectors.Dimensions[0]; dx += 1)
                        {
                            Detector detector = retina.Detectors[dx, dy];
                            double r = Math.Sqrt((detector.CenterXPixels - centerX) * (detector.CenterXPixels - centerX) + (detector.CenterYPixels - centerY) * (detector.CenterYPixels - centerY));
                            if (r < DetectorsVisibleRadiusPixels)
                                miniColumnDetectors.Add(detector);
                        }

                    MiniColumn miniColumn = new MiniColumn(
                        constants,
                        mcx,
                        mcy,
                        miniColumnDetectors,
                        centerX,
                        centerY);

                    MiniColumns[mcx, mcy] = miniColumn;

                    if (miniColumnR < 0.000001)
                        CenterMiniColumn = miniColumn;                        
                }
            }            

        HashSet<Detector> subArea_DetectorsHashSet = new(retina.Detectors.Dimensions[0] * retina.Detectors.Dimensions[1]);
        List<MiniColumn> subArea_MiniColumns = new List<MiniColumn>(constants.CalculationsSubAreaRadius_MiniColumns is not null ?
            constants.CalculationsSubArea_MiniColumns_Count :
            MiniColumns.Dimensions[0] * MiniColumns.Dimensions[1]);

        foreach (int mcy in Enumerable.Range(0, MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, MiniColumns.Dimensions[0]))
            {
                var mc = MiniColumns[mcx, mcy];
                if (mc is not null)
                {
                    subArea_MiniColumns.Add(mc);
                    foreach (var d in mc.Detectors)
                    {
                        subArea_DetectorsHashSet.Add(d);
                    }
                }
            }
        SubArea_MiniColumns = subArea_MiniColumns.ToArray();
        SubArea_Detectors = subArea_DetectorsHashSet.ToArray();

        //float sigma0 = constants.K3[0];
        //float sigma1 = constants.K3[1];

        // Находим ближайшие миниколонки для каждой миниколонки
        Parallel.For(
            fromInclusive: 0,
            toExclusive: SubArea_MiniColumns.Length,
            mci =>
            {
                MiniColumn mc = SubArea_MiniColumns[mci];

                //float k00 = GetNormalDistributionValue(sigma0, 0.0f);
                //float k01 = GetNormalDistributionValue(sigma1, 0.0f);
                //mc.K0 = (k00, k01);                
                mc.K_ForNearestMiniColumns.Add((constants.PositiveK[0], constants.NegativeK[0], mc));

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
                            //float k0 = GetNormalDistributionValue(sigma0, r);
                            //float k1 = GetNormalDistributionValue(sigma1, r);
                            float k0 = MathHelper.GetInterpolatedValue(constants.PositiveK, r);
                            float k1 = MathHelper.GetInterpolatedValue(constants.NegativeK, r);
                            mc.K_ForNearestMiniColumns.Add((k0, k1, nearestMc));
                        }
                    }
            });            

        VisualizationTableItems = new(1000);

        InputAutoencoder = new Autoencoder();
    }        

    /// <summary>
    ///     [0..MNISTImageWidth]
    /// </summary>
    public double DetectorsVisibleRadiusPixels { get; }

    public IConstants Constants { get; }

    //public float[] PositiveK;

    //public float[] NegativeK;

    public DenseMatrix<MiniColumn> MiniColumns;

    public MiniColumn? CenterMiniColumn { get; private set; }       

    public MiniColumn[] SubArea_MiniColumns { get; } = null!;
    public Detector[] SubArea_Detectors { get; } = null!;        
    public double SubArea_MiniColumns_Radius;

    public Autoencoder InputAutoencoder { get; } = null!;

    public List<VisualizationTableItem> VisualizationTableItems { get; }

    public Cortex_Simplified.MiniColumn? Temp_SuperActivityMax_MiniColumn
    {
        get;
        set;
    }

    public double Temp_WinnerMiniColumn_AverageGradientInPoint_Delta { get; set; }
    public double Temp_WinnerMiniColumn_AverageGradientInPoint_Magnitude { get; set; }

    public void GenerateOwnedData(Retina retina)
    {
        //InputAutoencoder.GenerateOwnedData(retina.Detectors.Dimensions[0], retina.Detectors.Dimensions[1], bottleneckK: 10.0f, inputSpotDiameterK: 10.0f);
        //InputAutoencoder.GenerateOwnedData(retina.Detectors.Data.Length, retina.Detectors.Data.Length / 10, null);
    }

    public void Prepare()
    {
        //InputAutoencoder.Prepare();
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        if (context as string == "autoencoders")
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(MiniColumns.Data.Length);
                for (int mci = 0; mci < MiniColumns.Data.Length; mci += 1)
                {
                    using (writer.EnterBlock(1))
                    {
                        MiniColumn miniColumn = MiniColumns.Data[mci];
                        writer.WriteOwnedDataSerializableAndRecreatable(miniColumn, context);
                    }
                }
            }
        }
        else if (context as string == "autoencoder")
        {
            using (writer.EnterBlock(1))
            {
                InputAutoencoder.SerializeOwnedData(writer, null);
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
                            using (Block block2 = reader.EnterBlock())
                            {
                                MiniColumn? miniColumn = MiniColumns.Data[mci];
                                if (miniColumn is not null)
                                    reader.ReadOwnedDataSerializableAndRecreatable<MiniColumn>(() => miniColumn, context);
                            }
                        }
                        break;
                }
            }
        }
        else if (context as string == "autoencoder")
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        InputAutoencoder.DeserializeOwnedData(reader, null);
                        break;
                }
            }
        }
    }        

    public class MiniColumn : IMiniColumn, IMiniColumnActivity, ISerializableModelObject
    {
        public MiniColumn(IConstants constants, int mcx, int mcy, List<Detector> detectors, double centerX, double centerY)
        {
            Constants = constants;
            Detectors = detectors;
            MCX = mcx;
            MCY = mcy;
            CenterX = centerX;
            CenterY = centerY;
            Temp_Hash = new float[constants.HashLength];                
            Memories = new(constants.MemoriesMaxCount);
            Temp_Memories = new(constants.MemoriesMaxCount);                       
            Temp_ShortHashConverted = new float[constants.ShortHashLength];                

            K_ForNearestMiniColumns = new FastList<(float, float, IMiniColumnActivity)>((int)(Math.PI * constants.SuperActivityRadius_MiniColumns * constants.SuperActivityRadius_MiniColumns) + 10);
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
        /// </summary>
        public (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Temp_Activity;

        /// <summary>
        ///     Текущая суммарная активность миниколонки с учетом активностей соседей при подаче примера
        /// </summary>
        public float Temp_SuperActivity;

        /// <summary>
        ///     Текущий хэш активных детекторов при подаче примера.
        /// </summary>
        public readonly float[] Temp_Hash;

        //public Color Temp_ActivityColor;

        //public Color Temp_SuperActivityColor;

        public bool Temp_IsSynced;

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

        (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) IMiniColumnActivity.Activity { get => Temp_Activity; set => Temp_Activity = value; }

        float IMiniColumnActivity.SuperActivity { get => Temp_SuperActivity; set => Temp_SuperActivity = value; }

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
        public MiniColumn? SelectedSuperActivityMax_MiniColumn;

        public float MaxActivity = float.MinValue;
        public readonly List<MiniColumn> ActivityMax_MiniColumns = new();

        public float MaxSuperActivity = float.MinValue;
        public readonly List<MiniColumn> SuperActivityMax_MiniColumns = new();
        public MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
        {
            if (SuperActivityMax_MiniColumns.Count == 0)
            {
                SelectedSuperActivityMax_MiniColumn = null;                
            }
            else if (SuperActivityMax_MiniColumns.Count == 1)
            {
                SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[0];                
            }
            else
            {
                SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[random.Next(SuperActivityMax_MiniColumns.Count)];
            }
            return SelectedSuperActivityMax_MiniColumn;
        }
    }
}    


//private float GetNormalDistributionValue(float sigma, float r)
//{
//    return (1.0f / (sigma * MathF.Sqrt(2.0f * MathF.PI))) * MathF.Exp(-0.5f * r * r / (sigma * sigma));
//}

//public class Cortex : IOwnedDataSerializable
//{
//    /// <summary>
//    ///     
//    /// </summary>
//    /// <param name="constants"></param>        
//    public Cortex(
//        ICortexConstants constants,
//        Retina retina)
//    {
//        Constants = constants;

//        Random random = new();

//        DetectorsVisibleRadius = Math.Sqrt(constants.MiniColumnVisibleDetectorsCount * constants.DetectorDelta * constants.DetectorDelta / Math.PI);

//        MiniColumns = new DenseTensor<MiniColumn>(constants.CortexWidth, constants.CortexHeight);
//        double minCenterX = DetectorsVisibleRadius;
//        double maxCenterX = constants.ImageWidth - DetectorsVisibleRadius;
//        double deltaCenterX = (maxCenterX - minCenterX) / (constants.CortexWidth - 1);
//        double minCenterY = DetectorsVisibleRadius;
//        double maxCenterY = constants.ImageHeight - DetectorsVisibleRadius;
//        double deltaCenterY = (maxCenterY - minCenterY) / (constants.CortexHeight - 1);                        

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

//                mc.K_ForNearestMiniColumns.Add(((1.0f, 1.0f), new List<MiniColumn>(8)));

//                for (int mcy = mc.MCY - 1; mcy < mc.MCY + 1; mcy += 1)
//                    for (int mcx = mc.MCX - 1; mcx < mc.MCX + 1; mcx += 1)
//                    {
//                        if (mcx < 0 ||
//                                mcx >= constants.CortexWidth ||
//                                mcy < 0 ||
//                                mcy >= constants.CortexHeight ||
//                                (mcx == mc.MCX && mcy == mc.MCY))
//                            continue;

//                        MiniColumn nearestMc = MiniColumns[mcx, mcy];

//                        mc.K_ForNearestMiniColumns[0].Item2.Add(nearestMc);
//                    }
//            });            
//    }

//    /// <summary>
//    ///     In big picture coordinates
//    /// </summary>
//    public double DetectorsVisibleRadius { get; }

//    public ICortexConstants Constants { get; }

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
//        public MiniColumn(ICortexConstants constants, int mcx, int mcy, List<Detector> detectors, double centerX, double centerY, Random random)
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

//            K_ForNearestMiniColumns = new List<((float, float), List<MiniColumn>)>();
//        }

//        public readonly ICortexConstants Constants;

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
//        public readonly List<((float, float), List<MiniColumn>)> K_ForNearestMiniColumns;

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

//    public interface ICortexConstants : IRetinaConstants
//    {
//        /// <summary>
//        ///     Количество миниколонок в зоне коры по оси X
//        /// </summary>
//        int CortexWidth { get; }

//        /// <summary>
//        ///     Количество миниколонок в зоне коры по оси Y
//        /// </summary>
//        int CortexHeight { get; }

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