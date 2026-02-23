#define GENERATE_INPUT_DATA

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using OpenCvSharp;
using Ssz.AI.Core;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using static Ssz.AI.Models.ImageProcessingModel.Cortex;

namespace Ssz.AI.Models.ImageProcessingModel;

public class Model01
{
    public const string FileName_Cortex = "ImageProcessingModel01_Cortex .bin";
    public const string FileName_StereoInput = "ImageProcessingModel01_StereoInput.bin";

    #region construction and destruction

    public Model01(Random random, bool onlyCeneterHypercolumn)
    {
        Random initialization_Random = new(6);

        Logger = new WrapperUserFriendlyLogger(
                new SszLogger("Ssz.AI.Models.ImageProcessingModel.Model01", "Ssz.AI.Models.ImageProcessingModel.Model01", new SszLoggerOptions()
                {
                    LogsDirectory = "Data",
#if DEBUG
                    LogFileName = "ImageProcessingModel01_Logs_Debug.txt"
#else
                    LogFileName = "ImageProcessingModel01_Logs.txt"
#endif
                }),
                new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));

        DataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();

        // Constants init.
        Rect2DFloat subImageRect = new Rect2DFloat(x: 0.45f, y: 0.45f, width: 0.1f, height: 0.1f);
        Constants.RetinaImagePixelSize = new PixelSize(
            (int)(Constants.RetinaImagePixelSize.Width * subImageRect.Width), 
            (int)(Constants.RetinaImagePixelSize.Height * subImageRect.Height));
        Constants.RetinaImageVerticalAngle = Constants.RetinaImageVerticalAngle * subImageRect.Height;

        LeftEye = CreateEye_ExceptRetina(pupil: new Vector3DFloat() { X = -Constants.DistanceBetweenEyes / 2, Y = 0.0f, Z = 0.0f }, subImageRect);
        RightEye = CreateEye_ExceptRetina(pupil: new Vector3DFloat() { X = Constants.DistanceBetweenEyes / 2, Y = 0.0f, Z = 0.0f }, subImageRect);

        GradientDistribution leftEye_GradientDistribution = new();
        GradientDistribution rightEye_GradientDistribution = new();

        StereoInput = new StereoInput();
#if GENERATE_INPUT_DATA

        (byte[] inputImagesLabels, byte[][] inputImageDatas, PixelSize inputImagesSize) = MNIST_Ex_Helper.ReadMNISTEx(
            labelsPath: @"Data\WriterInfo.npy",
            imagesPath: @"Data\Images(500x500).npy"
            );

        StereoInput.GenerateOwnedData(
                inputImagesSize,
                initialization_Random,
                Constants,
                leftEye_GradientDistribution,
                rightEye_GradientDistribution,
                inputImagesLabels,
                inputImageDatas,
                LeftEye,
                RightEye);
#else
        Helpers.SerializationHelper.LoadFromFileIfExists("StereoInput.bin", StereoInput, null);
#endif
        StereoInput.Prepare(); // Does nothing.

#if GENERATE_INPUT_DATA
        Helpers.SerializationHelper.SaveToFile("StereoInput.bin", StereoInput, null, null);
#endif

        LeftEye.Retina = new Retina(Constants);
        LeftEye.Retina.GenerateOwnedData(initialization_Random, Constants, leftEye_GradientDistribution);
        //Helpers.SerializationHelper.LoadFromFileIfExists("LeftEyeRetina.bin", LeftEye.Retina, null);
        LeftEye.Retina.Prepare();
        //Helpers.SerializationHelper.SaveToFile("LeftEyeRetina.bin", LeftEye.Retina, null);

        RightEye.Retina = new Retina(Constants);
        RightEye.Retina.GenerateOwnedData(initialization_Random, Constants, rightEye_GradientDistribution);
        //Helpers.SerializationHelper.LoadFromFileIfExists("RightEyeRetina.bin", RightEye.Retina, null);
        RightEye.Retina.Prepare();
        //Helpers.SerializationHelper.SaveToFile("RightEyeRetina.bin", RightEye.Retina, null);


        Cortex = new Cortex(Constants, Logger);
        Cortex.GenerateOwnedData(random, onlyCeneterHypercolumn);
        Cortex.Prepare();


        DataToDisplayHolder.GradientDistribution = leftEye_GradientDistribution;
    }

    #endregion

    #region public functions       

    public IUserFriendlyLogger Logger { get; }

    public DataToDisplayHolder DataToDisplayHolder = null!;

    public static readonly ModelConstants Constants = new();    

    public Eye LeftEye = null!;

    public Eye RightEye = null!;

    public StereoInput StereoInput = null!;

    public Cortex Cortex = null!;

    public StateInfo StateInfo = new();

    public VisualizationWithDesc[] GetImageWithDescs(
        Random random,
        double filterColorLow,
        double filterColorHigh)
    {
        var r1 = Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => (double)(mc.Temp_Activity.PositiveActivity + mc.Temp_Activity.NegativeActivity), valueMin: -1.0, valueMax: 1.0);
        var r2 = Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => mc.Temp_TotalEnergy);
        return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex, ii => ii.GradientAngleMagnitude_Color, filterColorLow, filterColorHigh)),
                    Desc = $"Воспоминания в миниколонках (Модуль и угол). Индекс вертушки: {GetPinwheelIndex(random, Cortex.MiniColumns, hypercolumnIndex: 0)}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex, ii => ii.RetinaXYAngle_Color)),
                    Desc = $"Воспоминания в миниколонках (XY)." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r1.Image),
                    Desc = $"Активность миниколонок; Min: {r1.ValueMin:F03}; Max: {r1.ValueMax:F03}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r2.Image),
                    Desc = $"Энергия (минимизируем); Min: {r2.ValueMin:F03}; Max: {r2.ValueMax:F03}" },                
            ];
    }    

    public void PutMemories_Pinwheel(Random random, int inMiniColumn_CortexMemoriesCount)
    {
        if (Cortex.MiniColumns is null)
            return;       

        var miniColumns = Cortex.MiniColumns;        
        for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)            
        {
            MiniColumn miniColumn = miniColumns[miniColumns_Index];
            MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.GetNearest_HyperColumnCenter_MiniColumn(miniColumn);
            //if (!nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMiniColumns.Contains(miniColumn))
            //    continue;

            InputItem inputItem = Cortex.AddInputItem(random, nearest_HyperColumnCenter_MiniColumn, miniColumn, miniColumn);                  

            var cortexMemory = Memory.FromInputItem(inputItem);
            
            for (int i = 0; i < inMiniColumn_CortexMemoriesCount; i += 1)
            {
                miniColumn.CortexMemories.Add(cortexMemory);
            }
        }
    }

    public void PutMemories_Random_SingleMemory(Random random, int cortexMemoriesCount)
    {
        if (Cortex.MiniColumns is null)
            return;

        var miniColumns = Cortex.MiniColumns;

        //for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)
        //{
        //    MiniColumn idealAngleMagnitude_MiniColumn = miniColumns[miniColumns_Index];
        //    MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.GetNearest_HyperColumnCenter_MiniColumn(idealAngleMagnitude_MiniColumn); 
        //    //if (!nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMiniColumns.Contains(idealAngleMagnitude_MiniColumn))
        //    //    continue;

        //    MiniColumn mainXY_MiniColumn = nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumn_MiniColumns
        //        [random.Next(nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumn_MiniColumns.Count)];

        //    InputItem inputItem = Cortex.AddInputItem(
        //        random,
        //        nearest_HyperColumnCenter_MiniColumn,
        //        idealAngleMagnitude_MiniColumn,
        //        mainXY_MiniColumn
        //        );
        //    var cortexMemory = Memory.FromInputItem(inputItem);

        //    var forMemoryMiniColumns = nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumn_MiniColumns.Where(mc => mc.CortexMemories.Count == 0).ToArray();            
        //    if (forMemoryMiniColumns.Length > 0)
        //    {
        //        var cortexMemories = forMemoryMiniColumns[random.Next(forMemoryMiniColumns.Length)].CortexMemories;                
        //        cortexMemories.Add(cortexMemory);
        //    }
        //}
    }

    public void PutMemories_Random_MultiMemory(Random random, int cortexMemoriesCount)
    {
        if (Cortex.MiniColumns is null || Cortex.HyperColumnCenters_MiniColumnIndices.Count == 0)
            return;        

        for (int miniColumns_Index = 0; miniColumns_Index < Cortex.MiniColumns.Count; miniColumns_Index += 1)
        {   
            var (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = GetRandomCortexMemory(random); 

            var forMemoryMiniColumns = nearest_HyperColumnCenter_MiniColumn.Temp_Strict_HyperColumn_MiniColumns;
            for (int i = 0; i < cortexMemoriesCount; i += 1)
            {
                var cortexMemories = forMemoryMiniColumns[random.Next(forMemoryMiniColumns.Count)].CortexMemories;                
                cortexMemories.Add(cortexMemory);
            }
        }
    }

    private (Memory, MiniColumn) GetRandomCortexMemory(Random random)
    {
        MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.MiniColumns[Cortex.HyperColumnCenters_MiniColumnIndices[random.Next(Cortex.HyperColumnCenters_MiniColumnIndices.Count)]];

        MiniColumn idealAngleMagnitude_MiniColumn = nearest_HyperColumnCenter_MiniColumn.Temp_Strict_HyperColumn_MiniColumns
            [random.Next(nearest_HyperColumnCenter_MiniColumn.Temp_Strict_HyperColumn_MiniColumns.Count)];

        InputItem inputItem = Cortex.AddInputItem(
            random,
            nearest_HyperColumnCenter_MiniColumn,
            idealAngleMagnitude_MiniColumn,
            nearest_HyperColumnCenter_MiniColumn
            );
        return (Memory.FromInputItem(inputItem), nearest_HyperColumnCenter_MiniColumn);
    }

    public async Task ProcessNAsync(float cortexMemoriesCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        if (Cortex.MiniColumns is null)
            return;

        var miniColumns = Cortex.MiniColumns;

        Stopwatch sw = Stopwatch.StartNew();

        for (int i = 0; i < cortexMemoriesCount * miniColumns.Count; i += 1)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = GetRandomCortexMemory(random);

            MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn(cortexMemory, random, cancellationToken, nearest_HyperColumnCenter_MiniColumn.Temp_SameFieldOfViewMiniColumns);
            bestForMemoryMiniColumn?.CortexMemories.Add(cortexMemory);

            if (sw.ElapsedMilliseconds > 1000)
            {
                await refreshAction();
                sw.Restart();
            }
        }

        await refreshAction();
    }

    //public async Task ReorderMemoriesAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    //{
    //    if (Constants.SingleMemory)
    //        await ReorderMemories_SingleMemoryAsync(random, cancellationToken, refreshAction, Cortex.MiniColumns);
    //    else
    //        await ReorderMemories_MultiMemoryAsync(random, cancellationToken, refreshAction, Cortex.MiniColumns);
    //}

    public async Task ReorderMemories_MultiMemoryAsync(
        Random random, 
        CancellationToken cancellationToken, 
        Func<Task> refreshAction, 
        FastList<MiniColumn> candidateMiniColumns, 
        int epochCount)
    {
        Stopwatch sw = Stopwatch.StartNew();

        int min_ChangesCount = Int32.MaxValue;
        int min_ChangesCount_UnchangedCount = 0;

        for (int epoch = 0; epoch < epochCount; epoch += 1)
        {
            int changedCount = 0;

            cancellationToken.ThrowIfCancellationRequested();

            FastList<(MiniColumn, int)> randomMiniColumnCortexMemoryIndices = new(candidateMiniColumns.Count * 10);
            for (int miniColumns_Index = 0; miniColumns_Index < candidateMiniColumns.Count; miniColumns_Index += 1)
            {
                var miniColumn = candidateMiniColumns[miniColumns_Index];

                for (int cortexMemory_Index = 0; cortexMemory_Index < miniColumn.CortexMemories.Count; cortexMemory_Index += 1)
                {
                    randomMiniColumnCortexMemoryIndices.Add((miniColumn, cortexMemory_Index));
                }
            }
            random.Shuffle(randomMiniColumnCortexMemoryIndices.Items);

            for (int index = 0; index < randomMiniColumnCortexMemoryIndices.Count; index += 1)
            {
                var it = randomMiniColumnCortexMemoryIndices[index];
                var miniColumn = it.Item1;
                Memory? cortexMemory = miniColumn.CortexMemories[it.Item2];
                if (cortexMemory is null)
                    continue;

                miniColumn.CortexMemories[it.Item2] = null;

                MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn(cortexMemory, random, cancellationToken, miniColumn.Temp_SameFieldOfViewMiniColumns);
                if (bestForMemoryMiniColumn is not null && !ReferenceEquals(bestForMemoryMiniColumn, miniColumn))
                {
                    bestForMemoryMiniColumn.CortexMemories.Add(cortexMemory);
                    changedCount += 1;
                }
                else
                {
                    miniColumn.CortexMemories[it.Item2] = cortexMemory;
                }

                if (refreshAction is not null && sw.ElapsedMilliseconds > 1000)
                {
                    await refreshAction();
                    sw.Restart();
                }
            }

            for (int mc_index = 0; mc_index < candidateMiniColumns.Count; mc_index += 1)
            {
                MiniColumn mc = candidateMiniColumns[mc_index];

                for (int mi = 0; mi < mc.CortexMemories.Count; mi += 1)
                {
                    Memory? cortexMemory = mc.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;

                    mc.Temp_CortexMemories.Add(cortexMemory);
                }

                mc.CortexMemories.Swap(mc.Temp_CortexMemories);
                mc.Temp_CortexMemories.Clear();
            }

            Logger.LogInformation($"Epoch: {epoch}/{epochCount};");

            if (refreshAction is not null)
                await refreshAction();

            if (changedCount < min_ChangesCount)
            {
                min_ChangesCount_UnchangedCount = 0;
                min_ChangesCount = changedCount;
            }
            else
            {
                min_ChangesCount_UnchangedCount += 1;
            }

            if (changedCount == 0 || min_ChangesCount_UnchangedCount > 1000)
                break;
        }

        Logger.LogInformation($"ReorderMemories Finished.");
    }

    public async Task AddNoizeAsync(int percents, Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        var randomMiniColumns = Cortex.MiniColumns.ToArray();        
        random.Shuffle(randomMiniColumns);
        randomMiniColumns = randomMiniColumns.Take(randomMiniColumns.Length * percents / 100).ToArray();

        for (int randomMiniColumns_Index = 0; randomMiniColumns_Index < randomMiniColumns.Length; randomMiniColumns_Index += 1)
        {
            var miniColumn = randomMiniColumns[randomMiniColumns_Index];

            MiniColumn adjacentMiniColumn = miniColumn.Temp_AdjacentMiniColumns[random.Next(miniColumn.Temp_AdjacentMiniColumns.Count)].Item2;

            miniColumn.CortexMemories.Swap(adjacentMiniColumn.CortexMemories);            
        }

        await refreshAction();
    }

    /// <summary>
    ///     
    /// </summary>
    /// <param name="random"></param>
    /// <param name="candidateMiniColumns"></param>
    /// <returns></returns>
    public float GetPinwheelIndex(Random random, FastList<MiniColumn> candidateMiniColumns, int hypercolumnIndex)
    {
        StateInfo stateInfo = new();
        stateInfo.MaxActivity = float.MinValue;
        stateInfo.ActivityMax_MiniColumns.Clear();
        
        for (int miniColumns_Index = 0; miniColumns_Index < candidateMiniColumns.Count; miniColumns_Index += 1)
        {
            var miniColumn = candidateMiniColumns[miniColumns_Index];

            var activity = MiniColumnsEnergyHelper.GetActivity(miniColumn, Cortex.IdealPinwheelMemories[hypercolumnIndex * 7], GetSimilarity, Constants);

            float a = activity.PositiveActivity + activity.NegativeActivity;
            if (a > stateInfo.MaxActivity)
            {
                stateInfo.MaxActivity = a;
                stateInfo.ActivityMax_MiniColumns.Clear();
                stateInfo.ActivityMax_MiniColumns.Add(miniColumn);
            }
            else if (a == stateInfo.MaxActivity)
            {
                stateInfo.ActivityMax_MiniColumns.Add(miniColumn);
            }
        };

        MiniColumn? pinwheelCenterMiniColumn = stateInfo.ActivityMax_MiniColumns.FirstOrDefault();

        return GetPinwheelIndex(pinwheelCenterMiniColumn, hypercolumnIndex);
    }

    public float GetPinwheelIndex(MiniColumn? pinwheelCenterMiniColumn, int hypercolumnIndex)
    {
        if (pinwheelCenterMiniColumn is null || pinwheelCenterMiniColumn.Temp_AdjacentMiniColumns.Count < 6)
            return 0.0f;

        float maxPinwheelIndex = Single.MinValue;
        for (int adjacentMiniColumns_StartIndex = 0; adjacentMiniColumns_StartIndex < 6; adjacentMiniColumns_StartIndex += 1)
        {
            float pinwheelIndex = 0.0f;
            for (int j = 1; j < 7; j += 1)
            {
                var idealPinwheelMemory = Cortex.IdealPinwheelMemories[hypercolumnIndex * 7 + j];

                MiniColumn miniColumn = pinwheelCenterMiniColumn.Temp_AdjacentMiniColumns[(adjacentMiniColumns_StartIndex + j) % 6].Item2;
                int cortexMemoriesCount = 0;
                float similaritySum = 0.0f;
                for (int mi = 0; mi < miniColumn.CortexMemories.Count; mi += 1)
                {
                    Memory? cortexMemory = miniColumn.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;
                    cortexMemoriesCount += 1;
                    similaritySum += (float)GetSimilarity(idealPinwheelMemory, cortexMemory);
                }
                if (cortexMemoriesCount > 0)
                    pinwheelIndex += similaritySum / cortexMemoriesCount;
            }
            if (pinwheelIndex > maxPinwheelIndex)
                maxPinwheelIndex = pinwheelIndex;
        }
        return maxPinwheelIndex;
    }

    public float GetEmptyMiniColumnsIndex(Random random, FastList<MiniColumn> candidateMiniColumns)
    {
        float emptyMinicolumnsIndex = 0.0f;
        for (int miniColumns_Index = 0; miniColumns_Index < candidateMiniColumns.Count; miniColumns_Index += 1)
        {
            var miniColumn = candidateMiniColumns[miniColumns_Index];

            if (miniColumn.CortexMemories.Count(cm => cm is not null) == 0)
                emptyMinicolumnsIndex -= 1.0f;
        }
        return emptyMinicolumnsIndex;
    }

    public void Flood(Random random, FastList<MiniColumn> candidateMiniColumns)
    {
        FastList<MiniColumn> excludeFromFlood_MiniColumns = new();

        for (int hypercolumnIndex = 0; hypercolumnIndex < Cortex.HyperColumnCenters_MiniColumnIndices.Count; hypercolumnIndex += 1)
        {
            //for (int candidateMiniColumns_Index = 0; candidateMiniColumns_Index < candidateMiniColumns.Count; candidateMiniColumns_Index += 1)
            //{
            //    var pinwheelCenterMiniColumn = candidateMiniColumns[candidateMiniColumns_Index];

            //    var activity = MiniColumnsEnergyHelper.GetActivity(pinwheelCenterMiniColumn, Cortex.IdealPinwheelMemories[hypercolumnIndex * 7], GetSimilarity, Constants);
            //    if (activity.PositiveActivity > 0.5f)
            //    {
            //        float pinwheelIndex = GetPinwheelIndex(pinwheelCenterMiniColumn, hypercolumnIndex);
            //        if (pinwheelIndex > 2.5f)
            //        {
            //            excludeFromFlood_MiniColumns.Add(pinwheelCenterMiniColumn);
            //            excludeFromFlood_MiniColumns.AddRange(pinwheelCenterMiniColumn.Temp_AdjacentMiniColumns.Select(it => it.Item2).ToArray());
            //        }
            //    }
            //}

            MiniColumn? pinwheelCenterMiniColumn = FindBestForMemoryMiniColumn(
                Cortex.IdealPinwheelMemories[hypercolumnIndex * 7],
                random,
                CancellationToken.None,
                candidateMiniColumns);
            if (pinwheelCenterMiniColumn is not null)
            {
                excludeFromFlood_MiniColumns.Add(pinwheelCenterMiniColumn);
                excludeFromFlood_MiniColumns.AddRange(pinwheelCenterMiniColumn.Temp_AdjacentMiniColumns.Select(it => it.Item2).ToArray());
            }
        }

        for (int candidateMiniColumns_Index = 0; candidateMiniColumns_Index < candidateMiniColumns.Count; candidateMiniColumns_Index += 1)
        {
            var candidateMiniColumn = candidateMiniColumns[candidateMiniColumns_Index];
            if (excludeFromFlood_MiniColumns.Any(it => ReferenceEquals(candidateMiniColumn, it)))
                continue;
            candidateMiniColumn.CortexMemories.Clear();
        }
    }

    #endregion

    #region private functions 

    private Eye CreateEye_ExceptRetina(Vector3DFloat pupil, Rect2DFloat subImageRect)
    {
        Eye eye = new();
        eye.Pupil = pupil;
        eye.RetinaUpperLeftXAngle = MathF.Atan2(Constants.PhysicalImageCenter.X - Constants.PhysicalImageSize.Width / 2 - pupil.X, Constants.PhysicalImageCenter.Z - pupil.Z);
        eye.RetinaUpperLeftYAngle = MathF.Atan2(Constants.PhysicalImageCenter.Y - Constants.PhysicalImageSize.Height / 2 - pupil.Y, Constants.PhysicalImageCenter.Z - pupil.Z);
        eye.RetinaBottomRightXAngle = MathF.Atan2(Constants.PhysicalImageCenter.X + Constants.PhysicalImageSize.Width / 2 - pupil.X, Constants.PhysicalImageCenter.Z - pupil.Z);
        eye.RetinaBottomRightYAngle = MathF.Atan2(Constants.PhysicalImageCenter.Y + Constants.PhysicalImageSize.Height / 2 - pupil.Y, Constants.PhysicalImageCenter.Z - pupil.Z);

        float widthAngle = eye.RetinaBottomRightXAngle - eye.RetinaUpperLeftXAngle;
        float heightAngle = eye.RetinaBottomRightYAngle - eye.RetinaUpperLeftYAngle;

        float subImageWidthAngle = widthAngle * subImageRect.Width;
        float subImageHeightAngle = heightAngle * subImageRect.Height;
        float subImageBiasXAngle = widthAngle * subImageRect.X;
        float subImageBiasYAngle = heightAngle * subImageRect.Y;

        eye.RetinaUpperLeftXAngle = eye.RetinaUpperLeftXAngle + subImageBiasXAngle;
        eye.RetinaUpperLeftYAngle = eye.RetinaUpperLeftYAngle + subImageBiasYAngle;
        eye.RetinaBottomRightXAngle = eye.RetinaUpperLeftXAngle + subImageWidthAngle;
        eye.RetinaBottomRightYAngle = eye.RetinaUpperLeftYAngle + subImageHeightAngle;

        return eye;
    }

    private MiniColumn? FindBestForMemoryMiniColumn(
        Memory cortexMemory, 
        Random random, 
        CancellationToken cancellationToken, 
        FastList<MiniColumn> candidateMiniColumns)
    {
        Parallel.For(
                fromInclusive: 0,
                toExclusive: candidateMiniColumns.Count,
                miniColumns_Index =>
                {
                    var miniColumn = candidateMiniColumns[miniColumns_Index];
                    
                    miniColumn.Temp_Activity = MiniColumnsEnergyHelper.GetActivity(miniColumn, cortexMemory, GetSimilarity, Constants);
                });

        StateInfo.MaxActivity = float.MinValue;
        StateInfo.ActivityMax_MiniColumns.Clear();

        if (Constants.TotalEnergyThreshold)
            StateInfo.MinTotalEnergy = Constants.K4;
        else
            StateInfo.MinTotalEnergy = float.MaxValue;
        StateInfo.TotalEnergyMin_MiniColumns.Clear();

        for (int miniColumns_Index = 0; miniColumns_Index < candidateMiniColumns.Count; miniColumns_Index += 1)
        {
            var miniColumn = candidateMiniColumns[miniColumns_Index];

            miniColumn.Temp_TotalEnergy = MiniColumnsEnergyHelper.GetTotalEnergy(miniColumn, Constants);

            float a = miniColumn.Temp_Activity.PositiveActivity + miniColumn.Temp_Activity.NegativeActivity;
            if (a > StateInfo.MaxActivity)
            {
                StateInfo.MaxActivity = a;
                StateInfo.ActivityMax_MiniColumns.Clear();
                StateInfo.ActivityMax_MiniColumns.Add(miniColumn);
            }
            else if (a == StateInfo.MaxActivity)
            {
                StateInfo.ActivityMax_MiniColumns.Add(miniColumn);
            }

            if (miniColumn.Temp_TotalEnergy < StateInfo.MinTotalEnergy)
            {
                StateInfo.MinTotalEnergy = miniColumn.Temp_TotalEnergy;
                StateInfo.TotalEnergyMin_MiniColumns.Clear();
                StateInfo.TotalEnergyMin_MiniColumns.Add(miniColumn);
            }
            else if (miniColumn.Temp_TotalEnergy == StateInfo.MinTotalEnergy)
            {
                StateInfo.TotalEnergyMin_MiniColumns.Add(miniColumn);
            }
        }

        return StateInfo.GetTotalEnergyMin_MiniColumn(random);
    }

    private async Task ReorderMemories_SingleMemoryAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction, FastList<MiniColumn> candidateMiniColumns)
    {
        Func<MiniColumn, FastList<(double, MiniColumn)>> getCandidateForSwapMiniColumns = mc => mc.Temp_AdjacentMiniColumns;

        int epochCount = 10000;

        for (int epoch = 0; epoch < epochCount; epoch += 1)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var randomMiniColumns = Cortex.MiniColumns.ToArray();
            random.Shuffle(randomMiniColumns);

            bool changed = false;

            for (int randomMiniColumns_Index = 0; randomMiniColumns_Index < randomMiniColumns.Length; randomMiniColumns_Index += 1)
            {
                var miniColumn = randomMiniColumns[randomMiniColumns_Index];
                if (miniColumn.CortexMemories.Count == 0)
                    continue;

                var candidateForSwapMiniColumns = getCandidateForSwapMiniColumns(miniColumn);

                miniColumn.Temp_MiniColumnEnergy = GetMiniColumnEnergy_SingleMemory(miniColumn);
                for (int i = 0; i < candidateForSwapMiniColumns.Count; i += 1)
                {
                    MiniColumn candidateForSwapMiniColumn = candidateForSwapMiniColumns[i].Item2;
                    candidateForSwapMiniColumn.Temp_MiniColumnEnergy = GetMiniColumnEnergy_SingleMemory(candidateForSwapMiniColumn);
                }

                double minEnergy = 0.0f;
                MiniColumn minEnergy_MiniColumn = miniColumn;

                for (int i = 0; i < candidateForSwapMiniColumns.Count; i += 1)
                {
                    MiniColumn candidateForSwapMiniColumn = candidateForSwapMiniColumns[i].Item2;

                    miniColumn.CortexMemories.Swap(candidateForSwapMiniColumn.CortexMemories);
                    double energy = -miniColumn.Temp_MiniColumnEnergy - candidateForSwapMiniColumn.Temp_MiniColumnEnergy
                        + GetMiniColumnEnergy_SingleMemory(miniColumn) + GetMiniColumnEnergy_SingleMemory(candidateForSwapMiniColumn);
                    miniColumn.CortexMemories.Swap(candidateForSwapMiniColumn.CortexMemories);

                    if (energy < minEnergy)
                    {
                        minEnergy = energy;
                        minEnergy_MiniColumn = candidateForSwapMiniColumn;
                    }
                }

                if (!ReferenceEquals(minEnergy_MiniColumn, miniColumn))
                {
                    miniColumn.CortexMemories.Swap(minEnergy_MiniColumn.CortexMemories);
                    changed = true;
                }
            }

            Logger.LogInformation($"Epoch: {epoch}/{epochCount};");
            await refreshAction();

            if (!changed)
                break;
        }
    }    

    private double GetMiniColumnEnergy_SingleMemory(MiniColumn miniColumn)
    {
        //if (miniColumn.CortexMemories.Count == 0)
        //    return 0.0;
        ////var d = MathHelper.NormalPdf(3.0f, 0.0f, 3.0f);

        //double energy = 0.0;
        //int count = 0;
        //for (int i = 0; i < miniColumn.Temp_K_2HyperColumnMiniColumns.Count; i += 1)
        //{
        //    var it = miniColumn.Temp_K_2HyperColumnMiniColumns[i];
        //    //energy -= MathHelper.NormalPdf(GetDistance(miniColumn.CortexMemories[0]!, it.Item2.CortexMemories[0]!), 0.0f, 3.0f);
        //    if (it.Item2.CortexMemories.Count > 0)
        //    {
        //        energy += Math.Log(GetSimilarity(miniColumn.CortexMemories[0]!, it.Item2.CortexMemories[0]!)) * it.Item1;
        //        if (Double.IsNaN(energy) || Double.IsInfinity(energy))
        //            throw new InvalidOperationException();
        //        count += 1;
        //    }
        //}
        //return energy / count;

        return 0.0;
    }

    private double GetSimilarity(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

        float hyperColumnDiameter_Retina2 = Cortex.Constants.MiniColumnFieldOfViewDiameter_Angle * Cortex.Constants.MiniColumnFieldOfViewDiameter_Angle;
        var r2 = (inpitItem1.HyperColumnCenter_RetinaXAngle - inpitItem2.HyperColumnCenter_RetinaXAngle) * (inpitItem1.HyperColumnCenter_RetinaXAngle - inpitItem2.HyperColumnCenter_RetinaXAngle)
            + (inpitItem1.HyperColumnCenter_RetinaYAngle - inpitItem2.HyperColumnCenter_RetinaYAngle) * (inpitItem1.HyperColumnCenter_RetinaYAngle - inpitItem2.HyperColumnCenter_RetinaYAngle);
        double k;
        if (r2 > hyperColumnDiameter_Retina2 * 1.5f)
            k = 0.0;
        else if (r2 > hyperColumnDiameter_Retina2 * 0.5f)
            k = 0.3; //0.00005; // 0.3;
        else
            k = 1.0;

        double radialDistance1 = inpitItem1.GradientMagnitude;
        double radialDistance2 = inpitItem2.GradientMagnitude;

        double gx1 = radialDistance1 * Math.Cos(inpitItem1.GradientAngle);
        double gy1 = radialDistance1 * Math.Sin(inpitItem1.GradientAngle);
        double gx2 = radialDistance2 * Math.Cos(inpitItem2.GradientAngle);
        double gy2 = radialDistance2 * Math.Sin(inpitItem2.GradientAngle);

        var gr2 = (gx1 - gx2) * (gx1 - gx2) + (gy1 - gy2) * (gy1 - gy2);

        double similarity = Math.Exp(-gr2 / 8.0) * k; // sigma == 2.0f

        if (similarity < 0.000001)
            similarity = 0.000001;
        //double activity = similarity - Cortex.Constants.K0;
        //activity = activity * k;
        //similarity = activity + Cortex.Constants.K0;
        //if (similarity < inpitItem1.SimilarityThreshold)
        //    return Single.NaN;

        return similarity;
    }

    private static float NormalPdf(float x2, float sigma)
    {
        // В знаменателе показателя экспоненты стоит 2 * sigma^2.
        float twoSigmaSquared = 2.0f * sigma * sigma;

        // Вычисляем показатель экспоненты:
        // exponent = - diff^2 / (2 * sigma^2).
        float exponent = -x2 / twoSigmaSquared;        

        return MathF.Exp(exponent);
    }    

    #endregion

    #region private fields    

    #endregion

    public class ModelConstants : ICortexConstants
    {
        public PixelSize RetinaImagePixelSize { get; set; } = new PixelSize(200, 200);

        public float RetinaImageVerticalAngle { get; set; } = MathHelper.DegreesToRadians(0.5f);

        public int GeneratedMinGradientMagnitude => 5;

        public int GeneratedMaxGradientMagnitude => 1200;

        public int MagnitudeRangesCount => 3;

        public double DetectorMinGradientMagnitude => 42;

        public Vector3DFloat PhysicalImageCenter { get; } = new Vector3DFloat() { X = 0.0f, Y = 0.0f, Z = 0.25f };

        public Size2DFloat PhysicalImageSize => new Size2DFloat(PhysicalImageCenter.Z * MathF.Sin(RetinaImageVerticalAngle), PhysicalImageCenter.Z * MathF.Sin(RetinaImageVerticalAngle));

        public float DistanceBetweenEyes => 0.064f;

        /// <summary>
        ///     Радиус гиперколонки в миниколонках.
        /// </summary>
        public int HyperColumnDefinedRadius_MiniColumns => 10;

        /// <summary>
        ///     Полное поле зрения (измеренное в миниколонках).
        ///     <para>При смечщении на такое число миниколонок, поле зрения смещается на 100%.</para>
        /// </summary>
        public int FullFieldOfView_MiniColumns => 20;

        public float MiniColumnFieldOfViewDiameter_Angle => MathHelper.DegreesToRadians(0.1f);

        /// <summary>
        ///     Количество детекторов, видимых одной миниколонкой
        /// </summary>
        public int MiniColumnVisibleDetectorsCount => 300;

        public int HashLength => 300;

        public int CortexWidth_MiniColumns => 100;

        public int CortexHeight_MiniColumns => 100;

        /// <summary>
        ///     Уровень подобия для нулевой активности
        /// </summary>
        public float K0 { get; set; } = 0.33f;

        /// <summary>
        ///     Уровень подобия с пустой миниколонкой.
        ///     Штраф за воспоминания (для равномерности заполнения).
        /// </summary>
        public float K2 { get; set; } = 1.0f; // Или чуть меньше, чем с точно таким же воспоминанием. Проверить что бы боьшая и малая вертушки не разрушались

        public float K3 { get; set; } = 0.20f;

        /// <summary>
        ///     Порог энергии
        /// </summary>
        public float K4 { get; set; } = -0.84f; // Чуть меньше, чем энергия пустого пространства

        public float[] PositiveK { get; set; } = [1.000f, 0.110f, 0.050f, 0.000f];

        public float[] NegativeK { get; set; } = [1.000f, 0.020f, 0.010f, 0.000f];

        /// <summary>
        ///     Включен ли порог энергии при накоплении воспоминаний
        /// </summary>
        public bool TotalEnergyThreshold { get; set; } = false;

        /// <summary>
        ///     Режим с одним воспоминанием в миниколонке.
        /// </summary>
        public bool SingleMemory { get; set; } = false;        
    }
}