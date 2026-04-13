#define GENERATE_INPUT_DATA
//#define SAVE_INPUT_DATA

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
using Ssz.AI.Models.MiniColumnDetailedModel;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Avalonia.Model3D;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using static Ssz.AI.Models.ImageProcessingModel.Cortex;

namespace Ssz.AI.Models.ImageProcessingModel;

public class Model01 : IDisposable
{
    public const string FileName_Cortex = "ImageProcessingModel01_Cortex.bin";
    public const string FileName_StereoInput = "ImageProcessingModel01_StereoInput.bin";

    #region construction and destruction

    public class Options
    {
        public bool OnlyCenterHyperColumn { get; set; }

        public bool LoadImagesSamplesFile { get; set; }
    }

    public Model01(Random random, Options options)
    {
        Random initialization_Random = new(7);

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

        //var value = ConfigurationHelper.GetValue<float>(Program.Host.Services.GetRequiredService<IConfiguration>(), Program.ConfigurationKey_Value, 0.0f);
        //Constants.DetectorFieldOfViewRadiusPixels = value;

        DataToDisplayHolder = DataToDisplayHolder.Instance;

        LeftEye = CreateEye_ExceptRetina(pupil: new Vector3DFloat() { X = -Constants.DistanceBetweenEyes / 2, Y = 0.0f, Z = 0.0f });
        LeftEye.IsRightEye = false;
        RightEye = CreateEye_ExceptRetina(pupil: new Vector3DFloat() { X = Constants.DistanceBetweenEyes / 2, Y = 0.0f, Z = 0.0f });
        RightEye.IsRightEye = true;

        GradientDistribution leftEye_GradientDistribution = new();
        GradientDistribution rightEye_GradientDistribution = new();

        StereoInput = new StereoInput();
#if GENERATE_INPUT_DATA

        byte[] inputImagesLabels; byte[][] inputImageDatas; PixelSize inputImagesSize;
        if (options.LoadImagesSamplesFile)            
            (inputImagesLabels, inputImageDatas, inputImagesSize) = MNIST_Ex_Helper.ReadMNISTEx(
                labelsPath: @"Data\WriterInfo.npy",
                imagesPath: @"Data\Images(500x500).npy"
                );
        else
            (inputImagesLabels, inputImageDatas, inputImagesSize) = (new byte[0], new byte[0][], new PixelSize(500, 500));

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
        Helpers.SerializationHelper.LoadFromFileIfExists("StereoInput.bin", StereoInput, null, Logger);
#endif
        StereoInput.Prepare(); // Does nothing.

#if GENERATE_INPUT_DATA && SAVE_INPUT_DATA
        Helpers.SerializationHelper.SaveToFile("StereoInput.bin", StereoInput, null, null);
#endif

        LeftEye.Retina = new Retina(Constants, Logger);
#if GENERATE_INPUT_DATA
        LeftEye.Retina.GenerateOwnedData(initialization_Random, leftEye_GradientDistribution);
#else
        Helpers.SerializationHelper.LoadFromFileIfExists("LeftEyeRetina.bin", LeftEye.Retina, null, Logger);
#endif
        LeftEye.Retina.Prepare();
#if GENERATE_INPUT_DATA && SAVE_INPUT_DATA
        Helpers.SerializationHelper.SaveToFile("LeftEyeRetina.bin", LeftEye.Retina, null, Logger);
#endif

        RightEye.Retina = new Retina(Constants, Logger);
#if GENERATE_INPUT_DATA
        RightEye.Retina.GenerateOwnedData(initialization_Random, rightEye_GradientDistribution);
#else
        Helpers.SerializationHelper.LoadFromFileIfExists("RightEyeRetina.bin", RightEye.Retina, null, Logger);
#endif
        RightEye.Retina.Prepare();
#if GENERATE_INPUT_DATA && SAVE_INPUT_DATA
        Helpers.SerializationHelper.SaveToFile("RightEyeRetina.bin", RightEye.Retina, null, Logger);
#endif


        Cortex = new Cortex(Constants, Logger);
        Cortex.GenerateOwnedData(initialization_Random, options.OnlyCenterHyperColumn);
        Cortex.Prepare(LeftEye, RightEye, random);

        // Optimization
        HashSet<RetinaPoint> toCalculateRetinaPoints = new();
        foreach (var d in Cortex.MiniColumns[Cortex.HyperColumnCenters_MiniColumnIndices[0]].Temp_LeftEye_Detectors)
        {
            foreach (var rp in d.Temp_RetinaPoints)
            {
                toCalculateRetinaPoints.Add(rp);
            }
        }
        LeftEye.Retina.Temp_ToCalculateRetinaPoints = new FastList<RetinaPoint>(toCalculateRetinaPoints.ToArray());

        DataToDisplayHolder.GradientDistribution = leftEye_GradientDistribution;
    }

    public void Dispose()
    {
        MiniColumnDetailed?.Dispose();
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

    public MiniColumnDetailed? MiniColumnDetailed;

    public StateInfo StateInfo = new();

    public VisualizationWithDesc[] GetImageWithDescs(
        Random random,
        double filterColorLow,
        double filterColorHigh)
    {
        var r0 = Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => (double)mc.GetMaxSomDistanceToAdjacent(), valueMin: 0.0, valueMax: 1.0);
        var r1 = Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => (double)(mc.Temp_Activity.PositiveActivity + mc.Temp_Activity.NegativeActivity), valueMin: -1.0, valueMax: 1.0);
        var r2 = Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => mc.Temp_TotalEnergy);

        Cortex.CalculateSomCortexMemories(random);

        return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Cortex.Temp_LastMiniColumn_SampleVisualisation?.FullImage),
                    Desc = $"Картинка." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Cortex.Temp_LastMiniColumn_SampleVisualisation?.Image),
                    Desc = $"Видимая миниколонкой картинка." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Cortex.Temp_LastMiniColumn_SampleVisualisation?.GradientImage),
                    Desc = $"Видимый миниколонкой градиент." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Cortex.Temp_LastMiniColumn_SampleVisualisation?.DetectorsActivationImage),
                    Desc = $"Видимые миниколонкой детекторы." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r2.Image),
                    Desc = $"Энергия (минимизируем); Min: {r2.ValueMin:F03}; Max: {r2.ValueMax:F03}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex, mc => mc.Temp_SomCortexMemories, ii => ii.GradientAngleMagnitude_Color, filterColorLow, filterColorHigh)),
                    Desc = $"SOM. Модуль и угол." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r0.Image),
                    Desc = $"SOM. Максимальное расстояние до соседей; Min: {r0.ValueMin:F03}; Max: {r0.ValueMax:F03}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex, mc => mc.CortexMemories, ii => ii.GradientAngleMagnitude_Color, filterColorLow, filterColorHigh)),
                    Desc = $"Воспоминания в миниколонках (Модуль и угол). Индекс вертушки: {GetPinwheelIndex(random, Cortex.MiniColumns, hypercolumnIndex: 0)}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex, mc => mc.CortexMemories, ii => ii.HyperColumnCenter_Color)),
                    Desc = $"Воспоминания в миниколонках (XY)." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r1.Image),
                    Desc = $"Активность миниколонок; Min: {r1.ValueMin:F03}; Max: {r1.ValueMax:F03}" },                               
            ];
    }

    public VisualizationWithDesc[] GetImageWithDescs_MiniColumnDetailed1(
        Random random)
    {
        var r2 = Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => mc.Temp_TotalEnergy);

        return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex, mc => mc.Temp_SomCortexMemories, ii => ii.GradientAngleMagnitude_Color)),
                    Desc = $"SOM. Модуль и угол." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(r2.Image),
                    Desc = $"Энергия (минимизируем); Min: {r2.ValueMin:F03}; Max: {r2.ValueMax:F03}" }                
            ];
    }

    public Model3DScene? GetImageWithDescs_MiniColumnDetailed2(
        Random random)
    {
        if (MiniColumnDetailed is null)
            return null;

        return Visualization3D.Get_MiniColumnDetailed_Model3DScene(MiniColumnDetailed);
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

            var idealCortexMemory = Cortex.GetIdealCortexMemory(
                random,                
                idealAngleMagnitude_MiniColumn: miniColumn,
                Constants.TestGradientWidthRelative,
                Constants.TestGradientPositionRelative,
                LeftEye,
                hyperColumnCenter_MiniColumn: nearest_HyperColumnCenter_MiniColumn,
                main_MiniColumn: nearest_HyperColumnCenter_MiniColumn);
            
            for (int i = 0; i < inMiniColumn_CortexMemoriesCount; i += 1)
            {
                miniColumn.CortexMemories.Add(idealCortexMemory);                
            }

            Array.Copy(idealCortexMemory.Hash, miniColumn.Temp_SomWeights, idealCortexMemory.Hash.Length);
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
            var (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = GetRandomIdealCortexMemory(random); 

            var forMemoryMiniColumns = nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnStrict_MiniColumns;
            for (int i = 0; i < cortexMemoriesCount; i += 1)
            {
                var cortexMemories = forMemoryMiniColumns[random.Next(forMemoryMiniColumns.Count)].CortexMemories;                
                cortexMemories.Add(cortexMemory);
            }
        }
    }

    private int _totalIterations;
    private int _currentIteration;

    public async Task ProcessSomNAsync(int? epochsCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction, bool isIdeal)
    {
        if (Cortex.MiniColumns is null)
            return;

        var miniColumns = Cortex.MiniColumns;

        Memory? cortexMemory = null;
        MiniColumn? nearest_HyperColumnCenter_MiniColumn = null;

        Stopwatch sw = Stopwatch.StartNew();

        for (int m_index = 0; m_index < Cortex.Temp_IdealPinwheelMemories.Count; m_index += 1)
        {
            var idealPinwheelMemory = Cortex.Temp_IdealPinwheelMemories[m_index];
            if (idealPinwheelMemory.Temp_SimilarMemories is null)
                idealPinwheelMemory.Temp_SimilarMemories = new FastList<(Memory, MiniColumn)>(StereoInput.StereoInputSamples.Length / Cortex.Temp_IdealPinwheelMemories.Count);
            else
                idealPinwheelMemory.Temp_SimilarMemories.Clear();
        }
        float cortexMemory_BitsCountAverage = 0;
        int sampleProcessedCount = 0;
        int inputSamplesCount;
        if (isIdeal)
            inputSamplesCount = 10000;
        else
            inputSamplesCount = StereoInput.StereoInputSamples.Length;
        for (int sample_Index = 0; sample_Index < inputSamplesCount; sample_Index += 1)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (isIdeal)
                (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = GetRandomIdealCortexMemory(random);
            else
                (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = GetCortexMemory(random, StereoInput.StereoInputSamples[sample_Index]);

            int cortexMemory_BitsCount = (int)TensorPrimitives.Sum(cortexMemory.Hash);
            cortexMemory_BitsCountAverage += cortexMemory_BitsCount;
            if (cortexMemory_BitsCount < 5)
                continue;

            Memory? idealPinwheelMemory_Best = Cortex.GetIdealPinwheelMemory_Best(cortexMemory.Hash, random);
            if (idealPinwheelMemory_Best is not null)
            {
                idealPinwheelMemory_Best.Temp_SimilarMemories!.Add((cortexMemory, nearest_HyperColumnCenter_MiniColumn));
                sampleProcessedCount += 1;
            }
        }
        if (inputSamplesCount > 0)
            cortexMemory_BitsCountAverage /= inputSamplesCount;
        Logger.LogInformation($"Samples: {sampleProcessedCount}/{inputSamplesCount}; cortexMemory_BitsCountAverage: {cortexMemory_BitsCountAverage};");

        float averageSamlesCount = (float)sampleProcessedCount / Cortex.Temp_IdealPinwheelMemories.Count(m => m.Temp_SimilarMemories!.Count > 0);
        int maxSamplesCount = (int)(averageSamlesCount * 1.5f);

        FastList<(Memory, MiniColumn)> memoriesToProcess = new FastList<(Memory, MiniColumn)>(inputSamplesCount);
        for (int m_index = 0; m_index < Cortex.Temp_IdealPinwheelMemories.Count; m_index += 1)
        {
            var idealPinwheelMemory = Cortex.Temp_IdealPinwheelMemories[m_index];
            if (idealPinwheelMemory.Temp_SimilarMemories!.Count > maxSamplesCount)
            {
                //random.Shuffle(idealPinwheelMemory.Temp_SimilarMemories.Items);
                memoriesToProcess.AddRange(idealPinwheelMemory.Temp_SimilarMemories!.Items.Slice(0, maxSamplesCount));
            }
            else
            {
                memoriesToProcess.AddRange(idealPinwheelMemory.Temp_SimilarMemories!.Items);
            }
        }

        if (epochsCount is null)
            epochsCount = (int)(12000.0f * 300.0f / memoriesToProcess.Count);

        if (epochsCount.Value == 0)
            _totalIterations += 1;
        else
            _totalIterations += epochsCount.Value * memoriesToProcess.Count;

        for (int epoch = 0; epoch < epochsCount; epoch += 1)
        {
            memoriesToProcess.Clear();
            for (int m_index = 0; m_index < Cortex.Temp_IdealPinwheelMemories.Count; m_index += 1)
            {
                var idealPinwheelMemory = Cortex.Temp_IdealPinwheelMemories[m_index];
                if (idealPinwheelMemory.Temp_SimilarMemories!.Count > maxSamplesCount)
                {
                    //random.Shuffle(idealPinwheelMemory.Temp_SimilarMemories.Items);
                    memoriesToProcess.AddRange(idealPinwheelMemory.Temp_SimilarMemories!.Items.Slice(0, maxSamplesCount));
                }
                else
                {
                    memoriesToProcess.AddRange(idealPinwheelMemory.Temp_SimilarMemories!.Items);
                }
            }
            random.Shuffle(memoriesToProcess.Items);

            for (int sample_Index = 0; sample_Index < memoriesToProcess.Count && _currentIteration < _totalIterations; sample_Index += 1)
            {
                cancellationToken.ThrowIfCancellationRequested();

                (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = memoriesToProcess[sample_Index];

                MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn_Som(
                    cortexMemory, 
                    random, 
                    cancellationToken, 
                    nearest_HyperColumnCenter_MiniColumn.Temp_SameFieldOfViewMiniColumns);
                //bestForMemoryMiniColumn?.CortexMemories.Add(cortexMemory);

                UpdateWeights(bestForMemoryMiniColumn, cortexMemory, _currentIteration, _totalIterations);

                //#if DEBUG
                //                MiniColumn? bestForMemoryMiniColumn2 = FindBestForMemoryMiniColumn_Som(cortexMemory, random, cancellationToken, nearest_HyperColumnCenter_MiniColumn.Temp_SameFieldOfViewMiniColumns);
                //#endif

                _currentIteration += 1;

                if (sw.ElapsedMilliseconds > 1000)
                {
                    Logger.LogInformation($"Epoch: {epoch}; Sample: {sample_Index}/{memoriesToProcess.Count};");

                    Cortex.Temp_LastMiniColumn_SampleVisualisation = GetImageVisualisation(cortexMemory);

                    await refreshAction();
                    sw.Restart();
                }
            }

            Logger.LogInformation($"Epoch: {epoch}; Sample: {memoriesToProcess.Count}/{memoriesToProcess.Count};");
        }

        if (nearest_HyperColumnCenter_MiniColumn is not null && cortexMemory is not null)
            Cortex.Temp_LastMiniColumn_SampleVisualisation = GetImageVisualisation(cortexMemory);

        await refreshAction();
    }

    public async Task CalculateTestMemoryWithSomAsync(Random random, CancellationToken cancellationToken)
    {
        if (Cortex.MiniColumns is null)
            return;

        await Task.Delay(0);

        MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.MiniColumns[Cortex.HyperColumnCenters_MiniColumnIndices[random.Next(Cortex.HyperColumnCenters_MiniColumnIndices.Count)]];

        Memory cortexMemory = Cortex.GetIdealCortexMemory(            
            MathHelper.DegreesToRadians(Constants.TestGradientAngleDegrees),
            Constants.TestGradientMagnitude,
            Constants.TestGradientWidthRelative,
            Constants.TestGradientPositionRelative,
            LeftEye,
            hyperColumnCenter_MiniColumn: nearest_HyperColumnCenter_MiniColumn,
            main_MiniColumn: nearest_HyperColumnCenter_MiniColumn);        

        MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn_Som(
            cortexMemory, 
            random, 
            cancellationToken, 
            nearest_HyperColumnCenter_MiniColumn.Temp_SameFieldOfViewMiniColumns);        
    }          

    /// <summary>
    /// Обновление весов нейронов после нахождения BMU
    /// </summary>
    private void UpdateWeights(MiniColumn? bestForMemoryMiniColumn, Memory cortexMemory, int currentIteration, int totalIterations)
    {
        if (bestForMemoryMiniColumn is null)
            return;

        float fraction = (float)currentIteration / totalIterations;

        const float alpha0 = 0.1f;    // α0
        const float alphaMin = 0.01f; // α_min   Recommended: 0.01     
        float ratio_Alpha = alphaMin / alpha0;
        float alpha = alpha0 * MathF.Pow(ratio_Alpha, fraction);

        //if (currentIteration > totalIterations * 0.9f)
        //{
        //    // Фаза заморозки
        //    alpha = 0.001f;
        //}
        //float phase = (float)epoch / totalEpochs;
        //if (phase < 0.3f)      // Фаза 1: Ordering
        //    return 0.5f;
        //else if (phase < 0.8f) // Фаза 2: Convergence  
        //    return 0.1f * MathF.Exp(-(phase - 0.3f) / 0.5f);
        //else                   // Фаза 3: Заморозка
        //    return 0.0005f;    // Постоянно очень малая

        const float sigma0 = 7.0f;    // σ0
        const float sigmaMin = 1.5f;  // σ_min        // Recommended: 1.0    
        float ratio_Sigma = sigmaMin / sigma0;
        float sigma = sigma0 * MathF.Pow(ratio_Sigma, fraction);                

        // Обновление весов всех нейронов с учетом функции соседства
        bestForMemoryMiniColumn.Temp_NearestMiniColumns2.Clear();
        for (int index = 0; index < bestForMemoryMiniColumn.Temp_NearestMiniColumns.Count; index += 1)
        {
            var it = bestForMemoryMiniColumn.Temp_NearestMiniColumns[index];
            float distance_MiniColumns_Squared = (float)it.Item1;
            
            float neighborhood = MathF.Exp(-distance_MiniColumns_Squared / (2.0f * sigma * sigma));

            if (neighborhood < 1e-5f) 
                continue; // мелкий порог, чтобы не считать лишнее

            if (it.Item2.Index != bestForMemoryMiniColumn.Index)
                bestForMemoryMiniColumn.Temp_NearestMiniColumns2.Add((neighborhood, it.Item2));            

            TensorPrimitives.Subtract(cortexMemory.Hash, it.Item2.Temp_SomWeights, it.Item2.Temp_SomWeightsDiff);
            TensorPrimitives.MultiplyAdd(it.Item2.Temp_SomWeightsDiff, alpha * neighborhood, it.Item2.Temp_SomWeights, it.Item2.Temp_NewSomWeights);
        }

        const float lambda = 0.8f;
        if (lambda != 0.0f && bestForMemoryMiniColumn.Temp_NearestMiniColumns2.Count > 0)
        {
            // Вычисляем сумму: Σ_{r' ≠ s} g(r', s) · (w_{r'} - v)
            // для каждой компоненты d
#pragma warning disable CS0162 // Unreachable code detected
            Array.Clear(bestForMemoryMiniColumn.Temp_SomWeightsCorrection);
#pragma warning restore CS0162 // Unreachable code detected
            float neighborhood_sum = 0.0f;
            for (int index = 0; index < bestForMemoryMiniColumn.Temp_NearestMiniColumns2.Count; index += 1)
            {
                var it = bestForMemoryMiniColumn.Temp_NearestMiniColumns2[index];

                float neighborhood = it.Item1;
                neighborhood_sum += neighborhood;

                TensorPrimitives.Subtract(it.Item2.Temp_SomWeights, bestForMemoryMiniColumn.Temp_SomWeights, it.Item2.Temp_SomWeightsDiff);
                TensorPrimitives.MultiplyAdd(it.Item2.Temp_SomWeightsDiff, neighborhood, bestForMemoryMiniColumn.Temp_SomWeightsCorrection, bestForMemoryMiniColumn.Temp_SomWeightsCorrection);
            }

            // Применяем: w_s += η · λ · correction
            TensorPrimitives.MultiplyAdd(bestForMemoryMiniColumn.Temp_SomWeightsCorrection, alpha * lambda / neighborhood_sum, bestForMemoryMiniColumn.Temp_NewSomWeights, bestForMemoryMiniColumn.Temp_NewSomWeights);
        }

        for (int index = 0; index < bestForMemoryMiniColumn.Temp_NearestMiniColumns2.Count; index += 1)
        {
            var it = bestForMemoryMiniColumn.Temp_NearestMiniColumns2[index];

            (it.Item2.Temp_SomWeights, it.Item2.Temp_NewSomWeights) = (it.Item2.Temp_NewSomWeights, it.Item2.Temp_SomWeights);
        }
        (bestForMemoryMiniColumn.Temp_SomWeights, bestForMemoryMiniColumn.Temp_NewSomWeights) = (bestForMemoryMiniColumn.Temp_NewSomWeights, bestForMemoryMiniColumn.Temp_SomWeights);
    }

    private (Memory, MiniColumn) GetRandomIdealCortexMemory(Random random)
    {
        MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.MiniColumns[Cortex.HyperColumnCenters_MiniColumnIndices[random.Next(Cortex.HyperColumnCenters_MiniColumnIndices.Count)]];

        MiniColumn idealAngleMagnitude_MiniColumn = nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnStrict_MiniColumns
            [random.Next(nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnStrict_MiniColumns.Count)];

        Cortex.Memory idealCortexMemory = Cortex.GetIdealCortexMemory(                        
            testGradientAngle: MathHelper.NormalizeAngle(2.0f * MathF.PI * random.NextSingle()),
            testGradientMagnitude: (float)(Constants.MinGradientMagnitudeInclusive + (Constants.MaxGradientMagnitudeExclusive - Constants.MinGradientMagnitudeInclusive) * random.NextSingle()),
            Constants.TestGradientWidthRelative,
            Constants.TestGradientPositionRelative,
            LeftEye,
            nearest_HyperColumnCenter_MiniColumn,
            main_MiniColumn: nearest_HyperColumnCenter_MiniColumn);
        return (idealCortexMemory, nearest_HyperColumnCenter_MiniColumn);
    }

    private (Memory, MiniColumn) GetCortexMemory(Random random, StereoInputSample stereoInputSample)
    {
        MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.MiniColumns[Cortex.HyperColumnCenters_MiniColumnIndices[random.Next(Cortex.HyperColumnCenters_MiniColumnIndices.Count)]];
        
        LeftEye.Retina.CalculateRetinaPoints(stereoInputSample.LeftEye_GradientMatrix);        
        
        FastList<Detector> detectors = nearest_HyperColumnCenter_MiniColumn.Temp_LeftEye_Detectors;        
        for (int d_index = 0; d_index < detectors.Count; d_index += 1)
        {
            var detector = detectors[d_index];
            detector.Temp_IsActivated = detector.CalculateIsActivated();            
        }
        
        Cortex.Memory cortexMemory = Cortex.GetCortexMemory(            
            stereoInputSample,
            LeftEye,
            hyperColumnCenter_MiniColumn: nearest_HyperColumnCenter_MiniColumn,
            main_MiniColumn: nearest_HyperColumnCenter_MiniColumn);
        return (cortexMemory, nearest_HyperColumnCenter_MiniColumn);
    }

    private (Memory, MiniColumn) GetTestCortexMemory(Random random)
    {
        MiniColumn nearest_HyperColumnCenter_MiniColumn = Cortex.MiniColumns[Cortex.HyperColumnCenters_MiniColumnIndices[random.Next(Cortex.HyperColumnCenters_MiniColumnIndices.Count)]];        

        Cortex.Memory cortexMemory = Cortex.GetIdealCortexMemory(
            MathHelper.DegreesToRadians(Constants.TestGradientAngleDegrees),
            Constants.TestGradientMagnitude,
            Constants.TestGradientWidthRelative,
            Constants.TestGradientPositionRelative,
            LeftEye,
            hyperColumnCenter_MiniColumn: nearest_HyperColumnCenter_MiniColumn,
            main_MiniColumn: nearest_HyperColumnCenter_MiniColumn);
        return (cortexMemory, nearest_HyperColumnCenter_MiniColumn);
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

            var (cortexMemory, nearest_HyperColumnCenter_MiniColumn) = GetRandomIdealCortexMemory(random);

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

            var activity = MiniColumnsEnergyHelper.GetActivity(miniColumn, Cortex.Temp_IdealPinwheelCenterMemories[hypercolumnIndex * 7], GetSimilarity, Constants);

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
                var idealPinwheelMemory = Cortex.Temp_IdealPinwheelCenterMemories[hypercolumnIndex * 7 + j];

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
                Cortex.Temp_IdealPinwheelCenterMemories[hypercolumnIndex * 7],
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

    public void Create_MiniColumnDetailed(Random random)
    {        
        Logger.LogInformation("=== Модель миниколонки коры мозга ===");
        Logger.LogInformation($"Аксонов: {MiniColumnDetailedModel.MiniColumnDetailed.PyramidalAxonsCount}");
        Logger.LogInformation($"Синапсов на аксон: {MiniColumnDetailedModel.MiniColumnDetailed.SynapsesPerAxon}");
        Logger.LogInformation($"Всего синапсов: {(long)MiniColumnDetailedModel.MiniColumnDetailed.PyramidalAxonsCount * MiniColumnDetailedModel.MiniColumnDetailed.SynapsesPerAxon:N0}");
        Logger.LogInformation($"");

        // ----------------------------------------------------------
        //  СОЗДАНИЕ МИНИКОЛОНКИ
        //  На ~200 аксонов × 10 000 синапсов = 2 000 000 синапсов.
        //  Построение занимает несколько секунд.
        // ----------------------------------------------------------
        Console.Write("Генерация миниколонки...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        MiniColumnDetailed = new MiniColumnDetailed(random);
        sw.Stop();
        Logger.LogInformation($" готово за {sw.ElapsedMilliseconds} мс.");
        Logger.LogInformation($"");
    }

    public void MiniColumnDetailedModel_Create3D(Random random)
    {
        if (MiniColumnDetailed is null)
            return;

        var (cortexMemory, nearest_HyperColumnCenter_MiniColum) = GetTestCortexMemory(random);

        bool log = false;

        Stopwatch? sw = null;
        if (log)
        {
            Logger.LogInformation($"Активных аксонов: {TensorPrimitives.Sum(cortexMemory.Hash)}");
            //Console.Write("Индексы активных аксонов: ");
            //foreach (int idx in activeIndices)
            //    Console.Write($"{idx} ");
            Logger.LogInformation($"");
            Logger.LogInformation($"");

            // ----------------------------------------------------------
            //  ПАРАМЕТРЫ ПОИСКА
            // ----------------------------------------------------------

            Logger.LogInformation($"Параметры поиска:");
            Logger.LogInformation($"  Радиус R = {Constants.ZoneRadiusUm} мкм");
            Logger.LogInformation($"  Минимум N = {Constants.ActivatedSynapsesCount} уникальных активных аксонов в зоне");
            Logger.LogInformation($"");

            // ----------------------------------------------------------
            //  ПОИСК АКТИВНЫХ ЗОН
            // ----------------------------------------------------------
            Console.Write("Поиск активных зон...");

            sw = System.Diagnostics.Stopwatch.StartNew();
        }
        
        MiniColumnDetailed.FindActiveZones(cortexMemory.Hash, Constants.ZoneRadiusUm, Constants.ActivatedSynapsesCount);
        var activeZones = MiniColumnDetailed.Temp_ThalamocorticalZones;        

        if (log && activeZones is not null)
        {
            sw!.Stop();

            Logger.LogInformation($" готово за {sw.ElapsedMilliseconds} мс.");
            Logger.LogInformation($"");

            // ----------------------------------------------------------
            //  ВЫВОД РЕЗУЛЬТАТОВ
            // ----------------------------------------------------------
            Logger.LogInformation($"Найдено активных зон: {activeZones.Count}");
            Logger.LogInformation($"");

            int displayCount = Math.Min(activeZones.Count, 10);
            for (int i = 0; i < displayCount; i += 1)
            {
                var z = activeZones[i];
                Logger.LogInformation(
                    $"  Зона {i + 1,3}: центр=({z.Center.X,7:F1}, {z.Center.Y,7:F1}, {z.Center.Z,7:F1}) мкм  " +
                    $"уникальных аксонов={z.UniqueAxonCount,3}  " +
                    $"аксоны=[{string.Join(",", z.ActiveAxonIndices)}]");
            }

            if (activeZones.Count > displayCount)
                Logger.LogInformation($"  ... и ещё {activeZones.Count - displayCount} зон.");
        }
    }

    #endregion

    #region private functions 

    private Eye CreateEye_ExceptRetina(Vector3DFloat pupil)
    {
        Eye eye = new();
        eye.Pupil = pupil;

        float retinaCenterXAbsoluteAngle = MathF.Atan2(Constants.PhysicalImageCenter.X - pupil.X, Constants.PhysicalImageCenter.Z - pupil.Z);
        float retinaCenterYAbsoluteAngle = MathF.Atan2(Constants.PhysicalImageCenter.Y - pupil.Y, Constants.PhysicalImageCenter.Z - pupil.Z);        

        eye.RetinaUpperLeftXAbsoluteAngle = retinaCenterXAbsoluteAngle - Constants.RetinaImageAngle / 2.0f;
        eye.RetinaUpperLeftYAbsoluteAngle = retinaCenterYAbsoluteAngle - Constants.RetinaImageAngle / 2.0f;
        eye.RetinaBottomRightXAbsoluteAngle = retinaCenterXAbsoluteAngle + Constants.RetinaImageAngle / 2.0f;
        eye.RetinaBottomRightYAbsoluteAngle = retinaCenterYAbsoluteAngle + Constants.RetinaImageAngle / 2.0f;

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

    private MiniColumn? FindBestForMemoryMiniColumn_Som(
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

                    miniColumn.Temp_SomActivity = Cortex.GetDistance(miniColumn.Temp_SomWeights, cortexMemory.Hash);
                });

        StateInfo.MinTotalEnergy = float.MaxValue;
        StateInfo.TotalEnergyMin_MiniColumns.Clear();

        for (int miniColumns_Index = 0; miniColumns_Index < candidateMiniColumns.Count; miniColumns_Index += 1)
        {
            var miniColumn = candidateMiniColumns[miniColumns_Index];

            miniColumn.Temp_TotalEnergy = miniColumn.Temp_SomActivity;

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

    private double GetSimilarity(Memory cortexMemory1, Memory cortexMemory2)
    {
        return TensorPrimitives.CosineSimilarity(cortexMemory1.Hash, cortexMemory2.Hash);
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

    private SampleVisualisation GetImageVisualisation(Models.ImageProcessingModel.Cortex.Memory cortexMemory)
    {
        SampleVisualisation imageVisualisation = new();

        float horizontal_Pixels_K = Constants.RetinaImagePixelSize.Width / Constants.RetinaImageAngle;
        float vertical_Pixels_K = Constants.RetinaImagePixelSize.Height / Constants.RetinaImageAngle;
        int centerX = (int)(Constants.RetinaImagePixelSize.Width / 2.0f + cortexMemory.Main_RetinaXAngle * horizontal_Pixels_K);
        int centerY = (int)(Constants.RetinaImagePixelSize.Height / 2.0f + cortexMemory.Main_RetinaYAngle * vertical_Pixels_K);
        double radius = Constants.FullFieldOfViewDiameter_MiniColumn_Angle * vertical_Pixels_K / 2.0f;

        if (cortexMemory.StereoInputSample_Index < StereoInput.StereoInputSamples.Length)
        {
            StereoInputSample stereoInputSample = StereoInput.StereoInputSamples[cortexMemory.StereoInputSample_Index];

            Bitmap retinaFullImage;
            DenseMatrix<GradientInPoint> gradientMatrix;
            if (cortexMemory.RetinaImageData_IsRightEye)
            {
                retinaFullImage = MNIST_Ex_Helper.GetBitmap(stereoInputSample.RightRetinaImageData, Constants.RetinaImagePixelSize.Width, Constants.RetinaImagePixelSize.Height);
                gradientMatrix = stereoInputSample.RightEye_GradientMatrix;
            }
            else
            {
                retinaFullImage = MNIST_Ex_Helper.GetBitmap(stereoInputSample.LeftRetinaImageData, Constants.RetinaImagePixelSize.Width, Constants.RetinaImagePixelSize.Height);
                gradientMatrix = stereoInputSample.LeftEye_GradientMatrix;
            }                       

            imageVisualisation.FullImage = retinaFullImage;

            imageVisualisation.Image = BitmapHelper.GetSubBitmap(retinaFullImage, centerX, centerY, radius);

            imageVisualisation.GradientImage = BitmapHelper.GetSubBitmap(Visualisation.GetGradientBitmap(gradientMatrix), centerX, centerY, radius);
        }

        imageVisualisation.DetectorsActivationImage = BitmapHelper.GetSubBitmap(
            Visualisation.GetBitmapFromActivatedDetectors(cortexMemory.Temp_DetectorsActivated, 
                Constants.RetinaImagePixelSize.Width,
                Constants.RetinaImagePixelSize.Height,
                ((IRetinaConstants)Constants).RetinaDetectorsDeltaPixels),
            (int)(centerX / ((IRetinaConstants)Constants).RetinaDetectorsDeltaPixels),
            (int)(centerY / ((IRetinaConstants)Constants).RetinaDetectorsDeltaPixels),
            (int)(radius / ((IRetinaConstants)Constants).RetinaDetectorsDeltaPixels));

        return imageVisualisation;
    }    

    #endregion

    #region private fields     

    #endregion

    public class ModelConstants : ICortexConstants
    {
        public PixelSize RetinaImagePixelSize { get; set; } = new PixelSize(100, 100); // Full image: new PixelSize(200, 200);

        public float RetinaImageAngle { get; set; } = MathHelper.DegreesToRadians(0.5f); // Full image: MathHelper.DegreesToRadians(0.5f);

        public int MaxGradientMagnitudeExclusive => 1200; // Исходя из гистограммы

        public double MinGradientMagnitudeInclusive => 20; // Исходя из гистограммы 20

        public float GradientMagnitudeDelta => 10;

        public float GradientAngleDegreeDelta => 10;

        public Vector3DFloat PhysicalImageCenter { get; } = new Vector3DFloat() { X = 0.0f, Y = 0.0f, Z = 0.25f };

        public Size2DFloat PhysicalImageSize => new Size2DFloat(2.0f * PhysicalImageCenter.Z * MathF.Tan(RetinaImageAngle / 2.0f), 2.0f * PhysicalImageCenter.Z * MathF.Tan(RetinaImageAngle / 2.0f));

        public float DistanceBetweenEyes => 0.064f;

        public float RetinaPointDeltaPixels { get; set; } = 0.2f;

        public float DetectorFieldOfViewRadiusPixels { get; set; } = 0.6f;

        /// <summary>
        ///     Радиус гиперколонки в миниколонках.
        /// </summary>
        public int HyperColumnDefinedRadius_MiniColumns => 10;

        /// <summary>
        ///     Полное поле зрения (измеренное в миниколонках).
        ///     <para>При смещении на такое число миниколонок, поле зрения смещается на 100%.</para>
        /// </summary>
        public int FullFieldOfView_MiniColumns => 40;

        public float FullFieldOfViewDiameter_MiniColumn_Angle => MathHelper.DegreesToRadians(1f / 60f);

        /// <summary>
        ///     Количество детекторов, видимых одной миниколонкой
        /// </summary>
        public int MiniColumnVisibleDetectorsCount { get; set; } = 300; //700;

        public int HashLength => 200;

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

        public float DetectorRange_MiniColumns { get; set; } = 5.2f; // Исходя из картинки TensorPrimitives.Distance

        public float TestGradientAngleDegrees { get; set; } = 0;

        public float TestGradientMagnitude { get; set; } = 600;

        public float TestGradientWidthRelative { get; set; } = 0.3f;

        public float TestGradientPositionRelative { get; set; } = 0.0f;

        /// <summary>
        ///     радиус зоны в мкм
        /// </summary>
        public float ZoneRadiusUm { get; set; } = 20.0f; // Pyramidal: 14.0f;

        /// <summary>
        ///     минимум N уникальных активных аксонов
        /// </summary>
        public int ActivatedSynapsesCount { get; set; } = 4;
    }
}


//float localizedLearningK = 1.0f;
//if (it.Item2.Temp_AdjacentMiniColumns.Count > 0)
//{
//    localizedLearningK = 0.0f;
//    for (int mc_Index = 0; mc_Index < it.Item2.Temp_AdjacentMiniColumns.Count; mc_Index += 1)
//    {
//        var adjacentMiniColumn = it.Item2.Temp_AdjacentMiniColumns[mc_Index].Item2;
//        localizedLearningK += TensorPrimitives.CosineSimilarity(it.Item2.Temp_SomWeights, adjacentMiniColumn.Temp_SomWeights);
//    }
//    localizedLearningK /= it.Item2.Temp_AdjacentMiniColumns.Count;
//    localizedLearningK = 100.0f * (1.0f - localizedLearningK);
//    if (localizedLearningK > 1.0f)
//        localizedLearningK = 1.0f;
//}