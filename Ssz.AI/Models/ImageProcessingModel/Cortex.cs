using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Ssz.AI.Models.ImageProcessingModel;

public interface ICortexConstants : IRetinaConstants
{
    int CortexWidth_MiniColumns { get; }

    int CortexHeight_MiniColumns { get; }    

    /// <summary>
    ///     Уровень подобия для нулевой активности
    /// </summary>
    float K0 { get; set; }

    /// <summary>
    ///     Уровень подобия с пустой миниколонкой.
    ///     Штраф за воспоминания (для равномерности заполнения).
    /// </summary>
    float K2 { get; set; }

    /// <summary>
    ///     Минимальное подобие для мини-вертушки.
    /// </summary>
    float K3 { get; set; }

    /// <summary>
    ///     Порог энергии
    /// </summary>
    float K4 { get; set; }

    float[] PositiveK { get; set; }

    float[] NegativeK { get; set; }

    /// <summary>
    ///     Включен ли порог энергии при накоплении воспоминаний
    /// </summary>
    bool TotalEnergyThreshold { get; set; }

    /// <summary>
    ///     Режим с одним воспоминанием в миниколонке.
    /// </summary>
    bool SingleMemory { get; set; }
}

public partial class Cortex : ISerializableModelObject
{
    /// <summary>
    ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
    /// </summary>
    /// <param name="constants"></param>        
    public Cortex(
        ICortexConstants constants, 
        ILogger logger)
    {
        Constants = constants;
        Logger = logger;        

        MiniColumn_XAngle_K = constants.FullFieldOfViewDiameter_MiniColumn_Angle / constants.FullFieldOfView_MiniColumns;
        MiniColumn_YAngle_K = constants.FullFieldOfViewDiameter_MiniColumn_Angle / constants.FullFieldOfView_MiniColumns;        
    }

    #region public functions

    public readonly ICortexConstants Constants;

    public readonly ILogger Logger;

    public readonly float MiniColumn_XAngle_K;

    public readonly float MiniColumn_YAngle_K;    

    public FastList<MiniColumn> MiniColumns { get; private set; } = null!;

    public FastList<int> HyperColumnCenters_MiniColumnIndices { get; private set; } = null!;

    /// <summary>
    ///     Первое воспоминеие нулевое в идеальной вертушке. Следующие 6 воспоминаний вокруг нулевого в идеальной вертушке.
    ///     И так для каждой гиперколонки.
    /// </summary>
    public FastList<Memory> Temp_IdealPinwheelCenterMemories { get; private set; } = null!;

    /// <summary>
    ///     Набор идеальных воспоминаний.
    /// </summary>
    public FastList<Memory> Temp_IdealPinwheelMemories { get; private set; } = null!;

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(Random initialization_Random, bool onlyCenterHypercolumn)
    {
        MiniColumns = new FastList<MiniColumn>(Constants.CortexWidth_MiniColumns * Constants.CortexHeight_MiniColumns);

        float delta_MCX = 1.0f;
        float delta_MCY = MathF.Sqrt(1.0f - 0.5f * 0.5f);

        if (onlyCenterHypercolumn)
        {
            float maxRadius = Constants.HyperColumnDefinedRadius_MiniColumns + 0.00001f;

            MiniColumn? global_CenterMiniColumn = null;

            for (int mcj = -(int)(Constants.HyperColumnDefinedRadius_MiniColumns / delta_MCY); mcj <= (int)(Constants.HyperColumnDefinedRadius_MiniColumns / delta_MCY); mcj += 1)
                for (int mci = -Constants.HyperColumnDefinedRadius_MiniColumns; mci <= Constants.HyperColumnDefinedRadius_MiniColumns; mci += 1)
                {
                    float mcx = mci + ((mcj % 2 == 0) ? 0.0f : 0.5f);
                    float mcy = mcj * delta_MCY;

                    float radius = MathF.Sqrt(mcx * mcx + mcy * mcy);
                    if (radius < maxRadius)
                    {
                        MiniColumn miniColumn = new MiniColumn(
                            Constants)
                        {
                            Index = MiniColumns.Count,
                            MCX = mcx,
                            MCY = mcy
                        };

                        miniColumn.GenerateOwnedData();

                        MiniColumns.Add(miniColumn);

                        if (radius < 0.00001f)
                            global_CenterMiniColumn = miniColumn;                        
                    }
                }

            HyperColumnCenters_MiniColumnIndices = new FastList<int>([ global_CenterMiniColumn!.Index ]);
        }
        else
        {
            float maxRadius = Math.Min(Constants.CortexWidth_MiniColumns / 2.0f, Constants.CortexHeight_MiniColumns / 2.0f) + 0.00001f;

            for (int mcj = -(int)(Constants.CortexHeight_MiniColumns / (2.0f * delta_MCY)); mcj <= (int)(Constants.CortexHeight_MiniColumns / (2.0f * delta_MCY)); mcj += 1)
                for (int mci = -(int)(Constants.CortexWidth_MiniColumns / 2.0f); mci <= (int)(Constants.CortexWidth_MiniColumns / 2.0f); mci += 1)
                {
                    float mcx = mci + ((mcj % 2 == 0) ? 0.0f : 0.5f);
                    float mcy = mcj * delta_MCY;

                    float radius = MathF.Sqrt(mcx * mcx + mcy * mcy);
                    if (radius < maxRadius)
                    {
                        MiniColumn miniColumn = new MiniColumn(
                            Constants)
                        {
                            Index = MiniColumns.Count,
                            MCX = mcx,
                            MCY = mcy
                        };

                        miniColumn.GenerateOwnedData();

                        MiniColumns.Add(miniColumn);
                    }
                }

            int x_Limit_MiniColumns = Constants.CortexWidth_MiniColumns / 2 - 1;
            int y_Limit_MiniColumns = Constants.CortexHeight_MiniColumns / 2 - 1;

            HyperColumnCenters_MiniColumnIndices = new FastList<int>(20);
            delta_MCX = Constants.HyperColumnDefinedRadius_MiniColumns * 2.0f;
            delta_MCY = MathF.Sqrt(3.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns);
            for (int mcj = -(int)(Constants.CortexHeight_MiniColumns / (2.0f * delta_MCY)); mcj <= (int)(Constants.CortexHeight_MiniColumns / (2.0f * delta_MCY)); mcj += 1)
                for (int mci = -(int)(Constants.CortexWidth_MiniColumns / (2.0f * delta_MCX)); mci <= (int)(Constants.CortexWidth_MiniColumns / (2.0f * delta_MCX)); mci += 1)
                {
                    bool r = mcj % 2 == 0;
                    if ((30000 + mci + (r ? 0 : 2)) % 3 == 2)
                        continue;

                    float mcx = (mci + (r ? 0.0f : 0.5f)) * delta_MCX;
                    float mcy = mcj * delta_MCY;

                    if (mcx <= -x_Limit_MiniColumns || mcx >= x_Limit_MiniColumns ||
                            mcy <= -y_Limit_MiniColumns || mcy >= y_Limit_MiniColumns)
                        continue;

                    float min_abs_r2 = Single.MaxValue;
                    MiniColumn? nearestMiniColumn = null;
                    for (int mc_index = 0; mc_index < MiniColumns.Count; mc_index += 1)
                    {
                        MiniColumn mc = MiniColumns[mc_index];

                        float abs_r2 = (mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY);

                        if (abs_r2 < min_abs_r2)
                        {
                            min_abs_r2 = abs_r2;
                            nearestMiniColumn = mc;
                        }
                    }

                    if (nearestMiniColumn is not null)
                    {
                        nearestMiniColumn.HyperColumn_Mci = mci;
                        nearestMiniColumn.HyperColumn_Mcj = mcj;
                        HyperColumnCenters_MiniColumnIndices.Add(nearestMiniColumn.Index);
                    }
                }
        }        

        var filteredHyperColumnCenters_MiniColumnIndices = new FastList<int>(20);
        foreach (int mc_index in HyperColumnCenters_MiniColumnIndices)
        {
            MiniColumn hyperColumnCenter_MiniColumn = MiniColumns[mc_index];            

            FastList<MiniColumn> adjacentMiniColumns = new(6);
            for (int mc_index2 = 0; mc_index2 < MiniColumns.Count; mc_index2 += 1)
            {
                if (mc_index2 == mc_index)
                    continue;

                MiniColumn nearestMc = MiniColumns[mc_index2];

                float r_Squared = (nearestMc.MCX - hyperColumnCenter_MiniColumn.MCX) * (nearestMc.MCX - hyperColumnCenter_MiniColumn.MCX) + 
                    (nearestMc.MCY - hyperColumnCenter_MiniColumn.MCY) * (nearestMc.MCY - hyperColumnCenter_MiniColumn.MCY);
                float r = MathF.Sqrt(r_Squared);                

                if (r < 1.00001f)
                    adjacentMiniColumns.Add(nearestMc);
            }

            if (adjacentMiniColumns.Count == 6)
                filteredHyperColumnCenters_MiniColumnIndices.Add(mc_index);
        }
        HyperColumnCenters_MiniColumnIndices.Swap(filteredHyperColumnCenters_MiniColumnIndices);
    }

    public void Prepare(Eye leftEye, Eye rightEye, Random initialization_Random)
    {
        float hyperColumnMaxRadius_MiniColumns = 2.0f * Constants.HyperColumnDefinedRadius_MiniColumns + 0.000001f;
        float nearestRadius_MiniColumns = Constants.HyperColumnDefinedRadius_MiniColumns / 2.0f;
        float sameFieldOfViewRadius_MiniColumns = Constants.FullFieldOfView_MiniColumns / 2.0f;

        for (int mc_index = 0; mc_index < MiniColumns.Count; mc_index += 1)            
        {
            MiniColumn miniColumn = MiniColumns[mc_index];
            miniColumn.Prepare(sameFieldOfViewRadius_MiniColumns, nearestRadius_MiniColumns);
            miniColumn.Temp_SomWeights = new float[Constants.HashLength];
            miniColumn.Temp_SomWeightsDiff = new float[Constants.HashLength];
            for (int bit_Index = 0; bit_Index < Constants.HashLength; bit_Index += 1)
            {
                // Инициализация малыми случайными значениями
                miniColumn.Temp_SomWeights[bit_Index] = initialization_Random.NextSingle(); // * 0.1f;
            }

            MiniColumn nearest_HyperColumnCenter_MiniColumn = GetNearest_HyperColumnCenter_MiniColumn(miniColumn);
            if (nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMax_MiniColumns is null)
                nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMax_MiniColumns = new FastList<MiniColumn>((int)(Math.PI * 25.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));
            if (nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnStrict_MiniColumns is null)
                nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnStrict_MiniColumns = new FastList<MiniColumn>((int)(Math.PI * 25.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));
            nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnStrict_MiniColumns.Add(miniColumn);

            float yPixels_K = Constants.RetinaImagePixelSize.Height / Constants.RetinaImageAngle;
            float detectorsVisibleRadiusPixels = Constants.FullFieldOfViewDiameter_MiniColumn_Angle * yPixels_K / 2.0f;            
            float centerXPixels = Constants.RetinaImagePixelSize.Width / 2.0f + miniColumn.MCX * MiniColumn_XAngle_K * yPixels_K;
            float centerYPixels = Constants.RetinaImagePixelSize.Height / 2.0f + miniColumn.MCY * MiniColumn_YAngle_K * yPixels_K;
            miniColumn.Temp_LeftEye_Detectors = new FastList<Detector>(Constants.MiniColumnVisibleDetectorsCount);
            for (int dJ = (int)((centerYPixels - detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels); dJ < (int)((centerYPixels + detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels) && dJ < leftEye.Retina.Detectors.Dimensions[1]; dJ += 1)
                for (int dI = (int)((centerXPixels - detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels); dI < (int)((centerXPixels + detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels) && dI < leftEye.Retina.Detectors.Dimensions[0]; dI += 1)
                {
                    if (dI < 0 || dJ < 0)
                        continue;

                    Detector detector = leftEye.Retina.Detectors[dI, dJ]!;
                    double rPixels = Math.Sqrt((detector.CenterXPixels - centerXPixels) * (detector.CenterXPixels - centerXPixels) + (detector.CenterYPixels - centerYPixels) * (detector.CenterYPixels - centerYPixels));
                    if (rPixels < detectorsVisibleRadiusPixels)
                        miniColumn.Temp_LeftEye_Detectors.Add(detector);
                }
            miniColumn.Temp_RightEye_Detectors = new FastList<Detector>(Constants.MiniColumnVisibleDetectorsCount);
            for (int dJ = (int)((centerYPixels - detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels); dJ < (int)((centerYPixels + detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels) && dJ < rightEye.Retina.Detectors.Dimensions[1]; dJ += 1)
                for (int dI = (int)((centerXPixels - detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels); dI < (int)((centerXPixels + detectorsVisibleRadiusPixels) / Constants.RetinaDetectorsDeltaPixels) && dI < rightEye.Retina.Detectors.Dimensions[0]; dI += 1)
                {
                    if (dI < 0 || dJ < 0)
                        continue;

                    Detector detector = rightEye.Retina.Detectors[dI, dJ]!;
                    double rPixels = Math.Sqrt((detector.CenterXPixels - centerXPixels) * (detector.CenterXPixels - centerXPixels) + (detector.CenterYPixels - centerYPixels) * (detector.CenterYPixels - centerYPixels));
                    if (rPixels < detectorsVisibleRadiusPixels)
                        miniColumn.Temp_RightEye_Detectors.Add(detector);
                }
        }

        // Находим ближайшие миниколонки для каждой миниколонки
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Count,
            (Action<int>)(mc_index =>
            {
                MiniColumn miniColumn = MiniColumns[mc_index];
                miniColumn.Temp_SameFieldOfViewMiniColumns.Add(miniColumn);
                miniColumn.Temp_NearestMiniColumns.Add((0.0, miniColumn));

                if (miniColumn.Temp_HyperColumnMax_MiniColumns is not null)
                    miniColumn.Temp_HyperColumnMax_MiniColumns.Add(miniColumn);

                for (int mc_index2 = 0; mc_index2 < MiniColumns.Count; mc_index2 += 1)                    
                {
                    if (mc_index2 == mc_index)                        
                        continue;

                    MiniColumn nearestMc = MiniColumns[mc_index2];                    

                    float r_Squared = (nearestMc.MCX - miniColumn.MCX) * (nearestMc.MCX - miniColumn.MCX) + (nearestMc.MCY - miniColumn.MCY) * (nearestMc.MCY - miniColumn.MCY);
                    float r = MathF.Sqrt(r_Squared);

                    if (r < 2.00001f)
                        miniColumn.Temp_K_SuperActivityMiniColumns.Add(
                            (MathHelper.GetInterpolatedValue(Constants.PositiveK, r),
                            MathHelper.GetInterpolatedValue(Constants.NegativeK, r),
                            nearestMc));

                    if (r < hyperColumnMaxRadius_MiniColumns)
                    {
                        //miniColumn.Temp_K_HyperColumnMiniColumns.Add((r, nearestMc));
                        if (miniColumn.Temp_HyperColumnMax_MiniColumns is not null)
                            miniColumn.Temp_HyperColumnMax_MiniColumns.Add(nearestMc);
                    }

                    //if (r < _2hypercolumnDefinedRadius_MiniColumns)
                    //    miniColumn.Temp_K_2HyperColumnMiniColumns.Add((r2, nearestMc));        

                    if (r < sameFieldOfViewRadius_MiniColumns)                    
                        miniColumn.Temp_SameFieldOfViewMiniColumns.Add(nearestMc);

                    if (r < 1.00001f)
                        miniColumn.Temp_AdjacentMiniColumns.Add((r, nearestMc));

                    if (r < nearestRadius_MiniColumns)
                        miniColumn.Temp_NearestMiniColumns.Add((r_Squared, nearestMc));
                }

                miniColumn.Temp_AdjacentMiniColumns = miniColumn.Temp_AdjacentMiniColumns
                    .OrderBy(it => MathF.Atan2(miniColumn.MCY - it.Item2.MCY, miniColumn.MCX - it.Item2.MCX))
                    .ToFastList();
            }));

        Temp_IdealPinwheelCenterMemories = new FastList<Memory>(HyperColumnCenters_MiniColumnIndices.Count * 7);
        Temp_IdealPinwheelMemories = new FastList<Memory>(MiniColumns.Count);
        foreach (int mc_index in HyperColumnCenters_MiniColumnIndices)
        {
            MiniColumn hyperColumnCenter_MiniColumn = MiniColumns[mc_index];

            var hyperColumnIdealPinwheelMemories = new FastList<Memory>(7);

            // Воспоминания для оценки качества вертушки TODO            
            hyperColumnIdealPinwheelMemories.Add(
                GetIdealCortexMemory(initialization_Random, hyperColumnCenter_MiniColumn, hyperColumnCenter_MiniColumn, hyperColumnCenter_MiniColumn, leftEye));

            foreach (var miniColumn in hyperColumnCenter_MiniColumn.Temp_HyperColumnMax_MiniColumns)
            {
                Temp_IdealPinwheelMemories.Add(
                    GetIdealCortexMemory(initialization_Random, hyperColumnCenter_MiniColumn, miniColumn, miniColumn, leftEye));
            }            

            foreach (var it in hyperColumnCenter_MiniColumn.Temp_AdjacentMiniColumns)
            {
                hyperColumnIdealPinwheelMemories.Add(
                    GetIdealCortexMemory(initialization_Random, hyperColumnCenter_MiniColumn, it.Item2, it.Item2, leftEye));
            }

            if (hyperColumnIdealPinwheelMemories.Count == 7)
                Temp_IdealPinwheelCenterMemories.AddRange(hyperColumnIdealPinwheelMemories.Items);
        }
    }

    public MiniColumn GetNearest_HyperColumnCenter_MiniColumn(MiniColumn miniColumn)
    {
        float min_abs_r2 = Single.MaxValue;
        MiniColumn? nearest_HyperColumnCenter_MiniColumn = null;
        for (int mc_index = 0; mc_index < HyperColumnCenters_MiniColumnIndices.Count; mc_index += 1)
        {
            MiniColumn mc = MiniColumns[HyperColumnCenters_MiniColumnIndices[mc_index]];

            float abs_r2 = (miniColumn.MCX - mc.MCX) * (miniColumn.MCX - mc.MCX) + (miniColumn.MCY - mc.MCY) * (miniColumn.MCY - mc.MCY);

            if (abs_r2 < min_abs_r2)
            {
                min_abs_r2 = abs_r2;
                nearest_HyperColumnCenter_MiniColumn = mc;
            }
        }
        return nearest_HyperColumnCenter_MiniColumn!;
    }

    /// <summary>
    ///     
    /// </summary>
    /// <param name="random"></param>
    /// <param name="hyperColumnCenter_MiniColumn"></param>
    /// <param name="idealAngleMagnitude_MiniColumn">For Angle and Magnitude</param>
    /// <param name="main_MiniColumn">For X_Retina and Y_Retina</param>
    /// <returns></returns>
    public Memory GetIdealCortexMemory(
        Random random, 
        MiniColumn hyperColumnCenter_MiniColumn, 
        MiniColumn idealAngleMagnitude_MiniColumn, 
        MiniColumn main_MiniColumn,
        Eye eye)
    {        
        float gradientAngle = MathHelper.NormalizeAngle(MathF.Atan2((idealAngleMagnitude_MiniColumn.MCY - hyperColumnCenter_MiniColumn.MCY), (idealAngleMagnitude_MiniColumn.MCX - hyperColumnCenter_MiniColumn.MCX)));
        if (MathF.Abs(hyperColumnCenter_MiniColumn.MCX) < 1.0f && MathF.Abs(hyperColumnCenter_MiniColumn.MCY) < 1.0f)
        {   
        }
        else
        {
            int mci = hyperColumnCenter_MiniColumn.HyperColumn_Mci;
            int mcj = hyperColumnCenter_MiniColumn.HyperColumn_Mcj;
            int mcj_remainder = (30000 + mcj) % 3;
            bool r = mcj % 2 == 0;
            int mci_remainder = (30000 + mci + (r ? 0 : 2)) % 3;
            bool c = mci_remainder == 0;            
            float angleHypercolumn;
            switch (mcj_remainder)
            {
                case 0:
                    angleHypercolumn = 0;
                    break;
                case 1:
                    angleHypercolumn = -MathF.PI * 2.0f / 3.0f;
                    break;
                case 2:
                    angleHypercolumn = MathF.PI * 2.0f / 3.0f;
                    break;
                default:
                    throw new InvalidOperationException();
            }
            gradientAngle = angleHypercolumn + (c ? gradientAngle : MathF.PI - gradientAngle);            
        }
        gradientAngle = MathHelper.NormalizeAngle(gradientAngle);

        float gradientMagnitude_K = Constants.MaxGradientMagnitudeExclusive / (Constants.HyperColumnDefinedRadius_MiniColumns);

        float gradientMagnitude = MathF.Sqrt((idealAngleMagnitude_MiniColumn.MCY - hyperColumnCenter_MiniColumn.MCY) * (idealAngleMagnitude_MiniColumn.MCY - hyperColumnCenter_MiniColumn.MCY)
            + (idealAngleMagnitude_MiniColumn.MCX - hyperColumnCenter_MiniColumn.MCX) * (idealAngleMagnitude_MiniColumn.MCX - hyperColumnCenter_MiniColumn.MCX)) * gradientMagnitude_K;

        Memory memory = new();

        memory.Hash = new float[Constants.HashLength];

        memory.GradientAngle = gradientAngle;
        memory.GradientMagnitude = gradientMagnitude;
        memory.Main_MiniColumnIndex = main_MiniColumn.Index;
        memory.Main_RetinaXAngle = main_MiniColumn.MCX * MiniColumn_XAngle_K;
        memory.Main_RetinaYAngle = main_MiniColumn.MCY * MiniColumn_YAngle_K;
              
        memory.HyperColumnCenter_MiniColumnIndex = hyperColumnCenter_MiniColumn!.Index;
        memory.HyperColumnCenter_RetinaXAngle = hyperColumnCenter_MiniColumn.MCX * MiniColumn_XAngle_K;
        memory.HyperColumnCenter_RetinaYAngle = hyperColumnCenter_MiniColumn.MCY * MiniColumn_YAngle_K;

        if (memory.GradientMagnitude < 40)
        {
            memory.GradientAngleMagnitude_Color = Color.White;
        }
        else
        {
            var gradientMagnitudeNormalized = memory.GradientMagnitude / Constants.MaxGradientMagnitudeExclusive;
            if (gradientMagnitudeNormalized > 1.0f)
                gradientMagnitudeNormalized = 1.0f;
            memory.GradientAngleMagnitude_Color = Visualisation.ColorFromHSV((double)(memory.GradientAngle + MathF.PI) / (2 * MathF.PI), gradientMagnitudeNormalized, 1.0);
        }

        float hyperColumnCenter_A = MathHelper.NormalizeAngle(MathF.Atan2(memory.HyperColumnCenter_RetinaYAngle, memory.HyperColumnCenter_RetinaXAngle));
        var hyperColumnCenter_M = MathF.Sqrt(hyperColumnCenter_MiniColumn.MCX * hyperColumnCenter_MiniColumn.MCX + hyperColumnCenter_MiniColumn.MCY * hyperColumnCenter_MiniColumn.MCY) * 2.0f /
            MathF.Sqrt(Constants.CortexWidth_MiniColumns * Constants.CortexWidth_MiniColumns + Constants.CortexHeight_MiniColumns * Constants.CortexHeight_MiniColumns);
        memory.HyperColumnCenter_Color = Visualisation.ColorFromHSV((double)(hyperColumnCenter_A + MathF.PI) / (2 * MathF.PI), hyperColumnCenter_M, 1.0);

        GradientInPoint gradientInPoint = new()
        {
            GradX = gradientMagnitude * Math.Cos(gradientAngle),
            GradY = gradientMagnitude * Math.Sin(gradientAngle),
            Magnitude = gradientMagnitude,
            Angle = gradientAngle
        };        
        FastList<Detector> detectors;
        if (eye.IsRightEye)
            detectors = main_MiniColumn.Temp_RightEye_Detectors;
        else
            detectors = main_MiniColumn.Temp_LeftEye_Detectors;
        for (int d_index = 0; d_index < detectors.Count; d_index += 1)
        {
            var detector = detectors[d_index];
            detector.CalculateIsActivated(eye.Retina, gradientInPoint, Constants);
            if (detector.Temp_IsActivated)
                memory.Hash[detector.BitIndexInHash] = 1.0f;
        }

        return memory;
    }

    /// <summary>
    ///     Precondition: !!! detectors must be Activated !!!
    /// </summary>
    /// <param name="eye"></param>
    /// <param name="stereoInputSample"></param>
    /// <param name="hyperColumnCenter_MiniColumn"></param>
    /// <param name="main_MiniColumn"></param>
    /// <returns></returns>
    public Memory GetCortexMemory(
        Eye eye, 
        StereoInputSample stereoInputSample,
        MiniColumn hyperColumnCenter_MiniColumn,
        MiniColumn main_MiniColumn)
    {
        Memory memory = new();

        memory.Hash = new float[Constants.HashLength];

        memory.StereoInputSample_Index = stereoInputSample.Index;
        memory.RetinaImageData_IsRightEye = eye.IsRightEye;       

        memory.Main_MiniColumnIndex = main_MiniColumn.Index;
        memory.Main_RetinaXAngle = main_MiniColumn.MCX * MiniColumn_XAngle_K;
        memory.Main_RetinaYAngle = main_MiniColumn.MCY * MiniColumn_YAngle_K;

        float averageGradientMagnitude_Sum = 0.0f;
        float averageGradientAngle_Sum = 0.0f;
        int activatedDetectorsCount = 0;
        FastList<Detector> detectors;
        if (eye.IsRightEye)
            detectors = main_MiniColumn.Temp_RightEye_Detectors;
        else
            detectors = main_MiniColumn.Temp_LeftEye_Detectors;
        for (int d_index = 0; d_index < detectors.Count; d_index += 1)
        {
            var detector = detectors[d_index];
            if (detector.Temp_IsActivated)
            {
                memory.Hash[detector.BitIndexInHash] = 1.0f;
                activatedDetectorsCount += 1;
                averageGradientMagnitude_Sum += detector.AverageGradientMagnitude;
                averageGradientAngle_Sum += detector.AverageGradientAngle;
            }
        }
        if (activatedDetectorsCount > 0)
        {
            averageGradientMagnitude_Sum /= activatedDetectorsCount;
            averageGradientAngle_Sum /= activatedDetectorsCount;
        }
        memory.GradientMagnitude = averageGradientMagnitude_Sum;
        memory.GradientAngle = averageGradientAngle_Sum;

        memory.HyperColumnCenter_MiniColumnIndex = hyperColumnCenter_MiniColumn!.Index;
        memory.HyperColumnCenter_RetinaXAngle = hyperColumnCenter_MiniColumn.MCX * MiniColumn_XAngle_K;
        memory.HyperColumnCenter_RetinaYAngle = hyperColumnCenter_MiniColumn.MCY * MiniColumn_YAngle_K;

        if (memory.GradientMagnitude < 40)
        {
            memory.GradientAngleMagnitude_Color = Color.White;
        }
        else
        {
            var gradientMagnitudeNormalized = memory.GradientMagnitude / Constants.MaxGradientMagnitudeExclusive;
            if (gradientMagnitudeNormalized > 1.0f)
                gradientMagnitudeNormalized = 1.0f;
            memory.GradientAngleMagnitude_Color = Visualisation.ColorFromHSV((double)(memory.GradientAngle + MathF.PI) / (2 * MathF.PI), gradientMagnitudeNormalized, 1.0);
        }

        float hyperColumnCenter_A = MathHelper.NormalizeAngle(MathF.Atan2(memory.HyperColumnCenter_RetinaYAngle, memory.HyperColumnCenter_RetinaXAngle));
        var hyperColumnCenter_M = MathF.Sqrt(hyperColumnCenter_MiniColumn.MCX * hyperColumnCenter_MiniColumn.MCX + hyperColumnCenter_MiniColumn.MCY * hyperColumnCenter_MiniColumn.MCY) * 2.0f /
            MathF.Sqrt(Constants.CortexWidth_MiniColumns * Constants.CortexWidth_MiniColumns + Constants.CortexHeight_MiniColumns * Constants.CortexHeight_MiniColumns);
        memory.HyperColumnCenter_Color = Visualisation.ColorFromHSV((double)(hyperColumnCenter_A + MathF.PI) / (2 * MathF.PI), hyperColumnCenter_M, 1.0);

        return memory;
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {            
            writer.WriteFastListOfOwnedDataSerializable(Temp_IdealPinwheelCenterMemories, context);
            writer.WriteFastListOfOwnedDataSerializable(Temp_IdealPinwheelMemories, context);
            writer.WriteFastListOfOwnedDataSerializable(MiniColumns, context);
            HyperColumnCenters_MiniColumnIndices.SerializeOwnedData(writer, context);            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:                    
                    Temp_IdealPinwheelCenterMemories = reader.ReadFastListOfOwnedDataSerializable(idx => new Memory(), context);
                    Temp_IdealPinwheelMemories = reader.ReadFastListOfOwnedDataSerializable(idx => new Memory(), context);
                    MiniColumns = reader.ReadFastListOfOwnedDataSerializable(idx => new MiniColumn(Constants), context);
                    HyperColumnCenters_MiniColumnIndices = new FastList<int>();
                    HyperColumnCenters_MiniColumnIndices.DeserializeOwnedData(reader, context);
                    break;
            }
        }
    }

    public void CalculateSomCortexMemories()
    {
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Count,
            (Action<int>)(mc_index =>
            {
                MiniColumn miniColumn = MiniColumns[mc_index];

                miniColumn.Temp_SomCortexMemories.Clear();

                float min = Single.MaxValue;
                Memory? idealPinwheelMemory_Best = null;
                for (int m_index = 0; m_index < Temp_IdealPinwheelMemories.Count; m_index += 1)
                {
                    var idealPinwheelMemory = Temp_IdealPinwheelMemories[m_index];
                    float f = TensorPrimitives.Distance(miniColumn.Temp_SomWeights, idealPinwheelMemory.Hash);
                    if (f < min)
                    {
                        min = f;
                        idealPinwheelMemory_Best = idealPinwheelMemory;
                    }    
                }

                if (idealPinwheelMemory_Best is not null)
                    miniColumn.Temp_SomCortexMemories.Add(idealPinwheelMemory_Best);
            }));
    }

    #endregion

    public class MiniColumn : ISerializableModelObject
    {
        public MiniColumn(ICortexConstants constants)
        {
            Constants = constants;
        }

        public readonly ICortexConstants Constants;

        public int Index;

        /// <summary>
        ///     Координата миниколонки по оси X (горизонтально вправо)
        /// </summary>
        public float MCX;

        /// <summary>
        ///     Координата миниколонки по оси Y (вертикально вниз)
        /// </summary>
        public float MCY;

        /// <summary>        
        ///     <para>Определено только для центров гиперколонок.</para>
        /// </summary>
        public int HyperColumn_Mci;

        /// <summary>        
        ///     <para>Определено только для центров гиперколонок.</para>
        /// </summary>
        public int HyperColumn_Mcj;

        /// <summary>
        ///     Окружающие миниколонки, для которых считается суперактивность.
        ///     <para>(k, MiniColumn)</para>        
        /// </summary>
        public FastList<(float PositiveK, float NegativeK, MiniColumn MiniColumn)> Temp_K_SuperActivityMiniColumns = null!;

        ///// <summary>
        /////     Окружающие миниколонки в радиусе примерно 0.5 гиперколонки.
        /////     <para>(r, MiniColumn)</para>        
        ///// </summary>
        //public FastList<(float, MiniColumn)> Temp_K_HyperColumnMiniColumns = null!;

        ///// <summary>
        /////     Окружающие миниколонки в радиусе примерно 1 гиперколонки.
        /////     <para>(r^2, MiniColumn)</para>        
        ///// </summary>
        //public FastList<(float, MiniColumn)> Temp_K_2HyperColumnMiniColumns = null!;

        public FastList<Detector> Temp_LeftEye_Detectors = null!;

        public FastList<Detector> Temp_RightEye_Detectors = null!;

        /// <summary>
        ///     <para>!!! Сама миниколонка !!! и окружающие миниколонки в радиусе примерно 2.0 * HyperColumnDefinedRadius_MiniColumns.</para>
        ///     <para>Круглой формы. Может пересекаться с другими гиперколонками.</para>
        ///     <para>Определено только для центров гиперколонок.</para>
        /// </summary>
        public FastList<MiniColumn> Temp_HyperColumnMax_MiniColumns = null!;

        /// <summary>
        ///     <para>!!! Сама миниколонка !!! и окружающие миниколонки в радиусе примерно HyperColumnDefinedRadius_MiniColumns.</para>
        ///     <para>Может быть неровной формы. Не пересекается с другими гиперколонками.</para>
        ///     <para>Определено только для центров гиперколонок.</para>
        /// </summary>
        public FastList<MiniColumn> Temp_HyperColumnStrict_MiniColumns = null!;

        /// <summary>
        ///     !!! Сама миниколонка !!! и окружающие миниколонки в радиусе примерно 2.0 * HyperColumnDefinedRadius_MiniColumns.
        /// </summary>
        public FastList<MiniColumn> Temp_SameFieldOfViewMiniColumns = null!;

        /// <summary>
        ///     Миниколонки - смежные соседи. Отсортированы по углу.
        ///     <para>(r, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_AdjacentMiniColumns = null!;

        /// <summary>
        ///     !!! Сама миниколонка !!! Миниколонки - ближайшие соседи в радиусе примерно 0.5 * HyperColumnDefinedRadius_MiniColumns.
        ///     <para>(r^2, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_NearestMiniColumns = null!;

        public double Temp_MiniColumnEnergy;

        public (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Temp_Activity;

        /// <summary>
        ///     Total system energy if memory in this minicolumn.
        /// </summary>
        public float Temp_TotalEnergy;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public FastList<Memory?> CortexMemories = null!;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public FastList<Memory?> Temp_CortexMemories = null!;

        /// <summary>
        ///     Визуализация SOM
        /// </summary>
        public FastList<Memory?> Temp_SomCortexMemories = null!;        

        public float[] Temp_SomWeights = null!;

        public float[] Temp_SomWeightsDiff = null!;

        public float Temp_SomActivity;

        public void GenerateOwnedData()
        {
            CortexMemories = new(10);            
        }

        public void Prepare(float sameFieldOfViewRadius_MiniColumns, float nearestRadius_MiniColumns)
        {
            Temp_K_SuperActivityMiniColumns = new FastList<(float, float, MiniColumn)>(18);
            //Temp_K_HyperColumnMiniColumns = new FastList<(float, MiniColumn)>((int)(Math.PI * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));
            //Temp_K_2HyperColumnMiniColumns = new FastList<(float, MiniColumn)>((int)(Math.PI * 4.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));            
            Temp_SameFieldOfViewMiniColumns = new FastList<MiniColumn>((int)(Math.PI * sameFieldOfViewRadius_MiniColumns * sameFieldOfViewRadius_MiniColumns + 5));
            Temp_AdjacentMiniColumns = new FastList<(double, MiniColumn)>(6);
            Temp_CortexMemories = new(10);
            Temp_SomCortexMemories = new(1);

            Temp_NearestMiniColumns = new FastList<(double, MiniColumn)>((int)(Math.PI * nearestRadius_MiniColumns * nearestRadius_MiniColumns + 5));
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(Index);
                writer.Write(MCX);
                writer.Write(MCY);
                writer.Write(HyperColumn_Mci);
                writer.Write(HyperColumn_Mcj);
                writer.WriteFastListOfOwnedDataSerializable(CortexMemories, context);                
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        Index = reader.ReadInt32();
                        MCX = reader.ReadSingle();
                        MCY = reader.ReadSingle();
                        HyperColumn_Mci = reader.ReadInt32();
                        HyperColumn_Mcj = reader.ReadInt32();
                        CortexMemories = reader.ReadFastListOfOwnedDataSerializable(idx => (Memory?)new Memory(), context);
                        break;                    
                }
            }
        } 
    }

    public class Memory : IOwnedDataSerializable
    {
        public float[] Hash = null!;

        public int StereoInputSample_Index;

        public bool RetinaImageData_IsRightEye;

        /// <summary>
        /// [-pi, pi)
        /// </summary>
        public float GradientAngle;

        /// <summary>
        /// 
        /// </summary>
        public float GradientMagnitude;

        public int Main_MiniColumnIndex;

        public float Main_RetinaXAngle;

        public float Main_RetinaYAngle;

        public int HyperColumnCenter_MiniColumnIndex;

        public float HyperColumnCenter_RetinaXAngle;

        public float HyperColumnCenter_RetinaYAngle;

        public Color GradientAngleMagnitude_Color;

        public Color HyperColumnCenter_Color;        

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            for (long i = 0; i < Hash.Length; i += 1)
            {
                writer.Write(Hash[i]);
            }
            writer.Write(StereoInputSample_Index);
            writer.Write(RetinaImageData_IsRightEye);
            writer.Write(GradientAngle);
            writer.Write(GradientMagnitude);
            writer.Write(Main_MiniColumnIndex);
            writer.Write(Main_RetinaXAngle);
            writer.Write(Main_RetinaYAngle);
            writer.Write(HyperColumnCenter_MiniColumnIndex);
            writer.Write(HyperColumnCenter_RetinaXAngle);
            writer.Write(HyperColumnCenter_RetinaYAngle);
            writer.Write(GradientAngleMagnitude_Color);
            writer.Write(HyperColumnCenter_Color);
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            for (long i = 0; i < Hash.Length; i += 1)
            {
                Hash[i] = reader.ReadSingle();
            }
            StereoInputSample_Index = reader.ReadInt32();
            RetinaImageData_IsRightEye = reader.ReadBoolean();            
            GradientAngle = reader.ReadSingle();
            GradientMagnitude = reader.ReadSingle();
            Main_MiniColumnIndex = reader.ReadInt32();
            Main_RetinaXAngle = reader.ReadSingle();
            Main_RetinaYAngle = reader.ReadSingle();
            HyperColumnCenter_MiniColumnIndex = reader.ReadInt32();
            HyperColumnCenter_RetinaXAngle = reader.ReadSingle();
            HyperColumnCenter_RetinaYAngle = reader.ReadSingle();
            GradientAngleMagnitude_Color = reader.ReadColor();
            HyperColumnCenter_Color = reader.ReadColor();
        }

        public override string ToString()
        {
            return $"Angle: {GradientAngle:F1}; Magnitude: {GradientMagnitude:F03};";
        }
    }
}


        ///// <summary>
        /////     Distance from center in ideal pinwheel in minicolumns
        ///// </summary>
        //public float DistanceFromCenter_MiniColumns = Single.MinValue;