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
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using OpenCvSharp;
using Ssz.AI.Core;
using Ssz.AI.Helpers;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using static Ssz.AI.Models.CortexVisualisationModel.Cortex;

namespace Ssz.AI.Models.CortexVisualisationModel;

public class Model02
{
    public const string FileName_Cortex = "CortexVisualisationModel_Cortex.bin";

    #region construction and destruction

    public Model02()
    {
        LoggersSet = new LoggersSet(
            NullLogger.Instance,
            new WrapperUserFriendlyLogger(
                new SszLogger("Ssz.AI.Models.CortexVisualisationModel.Model01", "Ssz.AI.Models.CortexVisualisationModel.Model01", new SszLoggerOptions()
                {
                    LogsDirectory = "Data",
#if DEBUG
                    LogFileName = "CortexVisualisationModel_Model01_Logs_Debug.txt"
#else
                    LogFileName = "CortexVisualisationModel_Model01_Logs.txt"
#endif
                }),
            new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s))));
    }

    #endregion

    #region public functions       

    public ILoggersSet LoggersSet { get; }

    public static readonly ModelConstants Constants = new();

    public Cortex Cortex = null!;

    public ActivitiyMaxInfo ActivitiyMaxInfo = new();

    public readonly Cortex.Memory[] PinwheelIndexConstantCortexMemories = new Cortex.Memory[7];

    public VisualizationWithDesc[] GetImageWithDescs(Random random)
    {        
        return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex)),
                    Desc = $"Воспоминания в миниколонках. Индекс вертушки: {GetPinwheelIndex(random, Cortex.MiniColumns)}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsValue(Cortex, 
                        (MiniColumn mc) => (double)(mc.Temp_Activity.PositiveActivity + mc.Temp_Activity.NegativeActivity), valueMin: -1.0, valueMax: 1.0)),
                    Desc = $"Активность" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsValue(Cortex,
                        (MiniColumn mc) => mc.Temp_SuperActivity, valueMin: -2.0, valueMax: 2.0)),
                    Desc = $"Суперактивность" }
            ];
    }    

    public void PutInitialMemoriesPinwheel(Random random, bool isRandom)
    {
        if (Cortex.MiniColumns is null)
            return;       

        var miniColumns = Cortex.MiniColumns;
        var randomMiniColumns = Cortex.MiniColumns.ToArray();
        random.Shuffle(randomMiniColumns);
        
        for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)            
        {
            MiniColumn miniColumn = miniColumns[miniColumns_Index];            

            InputItem inputItem = Cortex.AddInputItem(random, miniColumn);                  

            var cortexMemory = new Memory
            {
                InputItemIndex = inputItem.Index
            };

            float r = MathF.Sqrt(miniColumn.MCX * miniColumn.MCX + miniColumn.MCY * miniColumn.MCY) + 0.5f;
            int count = (int)((Constants.CortexRadius_MiniColumns + 0.5) / r);

            for (int i = 0; i < count; i += 1)
            {
                if (isRandom)
                {
                    randomMiniColumns[random.Next(randomMiniColumns.Length)].CortexMemories.Add(cortexMemory);
                }
                else
                {
                    miniColumn.CortexMemories.Add(cortexMemory);
                }
            }
        }
    }

    public async Task ProcessNAsync(int inputItemsCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        for (int i = 0; i < inputItemsCount; i += 1)
        {
            Memory cortexMemory = CreateMemory(random);

            MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn(cortexMemory, random, cancellationToken, Cortex.MiniColumns);
            bestForMemoryMiniColumn?.CortexMemories.Add(cortexMemory);

            if (i % 300 == 0)
                await refreshAction();
        }

        await refreshAction();
    }

    public async Task ReorderMemoriesAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        await ReorderMemoriesAsync(random, cancellationToken, refreshAction, Cortex.MiniColumns);
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

    public float GetPinwheelIndex(Random random, FastList<MiniColumn> candidateMiniColumns)
    {
        MiniColumn? centerMiniColumn = FindBestForMemoryMiniColumn(
            Memory.IdealPinwheelCenterMemory,
            random,
            CancellationToken.None,
            candidateMiniColumns);
        if (centerMiniColumn is null || centerMiniColumn.Temp_AdjacentMiniColumns.Count < 6)
            return 0.0f;

        float maxPinwheelIndex = Single.MinValue;
        for (int adjacentMiniColumns_StartIndex = 0; adjacentMiniColumns_StartIndex < 6; adjacentMiniColumns_StartIndex += 1)
        {            
            float pinwheelIndex = 0.0f;
            for (int j = 0; j < 6; j += 1)
            {
                var idealPinwheelMemory = Memory.IdealPinwheelMemories[j];

                MiniColumn miniColumn = centerMiniColumn.Temp_AdjacentMiniColumns[(adjacentMiniColumns_StartIndex + j) % 6].Item2;
                int cortexMemoriesCount = 0;
                float similaritySum = 0.0f;
                for (int mi = 0; mi < miniColumn.CortexMemories.Count; mi += 1)
                {
                    Memory? cortexMemory = miniColumn.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;
                    cortexMemoriesCount += 1;
                    similaritySum += GetSimilarity(idealPinwheelMemory, cortexMemory);
                }
                if (cortexMemoriesCount > 0)
                    pinwheelIndex += similaritySum / cortexMemoriesCount;
            }
            if (pinwheelIndex > maxPinwheelIndex)
                maxPinwheelIndex = pinwheelIndex;
        }
        return maxPinwheelIndex;
    }

    #endregion

    #region private functions       

    private Memory CreateMemory(Random random)
    {
        MiniColumn miniColumn = Cortex.MiniColumns[random.Next(Cortex.MiniColumns.Count)];

        InputItem inputItem = Cortex.AddInputItem(random, miniColumn);

        return new Memory
        {
            InputItemIndex = inputItem.Index
        };
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
                    
                    miniColumn.Temp_Activity = MiniColumnsActivityHelper.GetActivity(miniColumn, cortexMemory, GetSimilarity, Constants);
                });

        ActivitiyMaxInfo.MaxActivity = float.MinValue;
        ActivitiyMaxInfo.ActivityMax_MiniColumns.Clear();

        if (Constants.SuperactivityThreshold)
            ActivitiyMaxInfo.MaxSuperActivity = Constants.K4;
        else
            ActivitiyMaxInfo.MaxSuperActivity = float.MinValue;
        ActivitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();

        for (int miniColumns_Index = 0; miniColumns_Index < candidateMiniColumns.Count; miniColumns_Index += 1)
        {
            var miniColumn = candidateMiniColumns[miniColumns_Index];

            miniColumn.Temp_SuperActivity = MiniColumnsActivityHelper.GetSuperActivity(miniColumn, Constants);

            float a = miniColumn.Temp_Activity.PositiveActivity + miniColumn.Temp_Activity.NegativeActivity;
            if (a > ActivitiyMaxInfo.MaxActivity)
            {
                ActivitiyMaxInfo.MaxActivity = a;
                ActivitiyMaxInfo.ActivityMax_MiniColumns.Clear();
                ActivitiyMaxInfo.ActivityMax_MiniColumns.Add(miniColumn);
            }
            else if (a == ActivitiyMaxInfo.MaxActivity)
            {
                ActivitiyMaxInfo.ActivityMax_MiniColumns.Add(miniColumn);
            }

            if (miniColumn.Temp_SuperActivity > ActivitiyMaxInfo.MaxSuperActivity)
            {
                ActivitiyMaxInfo.MaxSuperActivity = miniColumn.Temp_SuperActivity;
                ActivitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
                ActivitiyMaxInfo.SuperActivityMax_MiniColumns.Add(miniColumn);
            }
            else if (miniColumn.Temp_SuperActivity == ActivitiyMaxInfo.MaxSuperActivity)
            {
                ActivitiyMaxInfo.SuperActivityMax_MiniColumns.Add(miniColumn);
            }
        }

        return ActivitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
    }

    private async Task ReorderMemoriesAsync(Random random, CancellationToken cancellationToken, Func<Task> refreshAction, FastList<MiniColumn> candidateMiniColumns)
    {
        int min_ChangesCount = Int32.MaxValue;
        int min_ChangesCount_UnchangedCount = 0;

        int epochCount = 100;

        for (int epoch = 0; epoch < epochCount; epoch += 1)
        {
            int changedCount = 0;

            for (int miniEpoch = 0; miniEpoch < 50; miniEpoch += 1)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var randomMiniColumns = candidateMiniColumns.ToArray();
                random.Shuffle(randomMiniColumns);                

                for (int randomMiniColumns_Index = 0; randomMiniColumns_Index < randomMiniColumns.Length; randomMiniColumns_Index += 1)
                {
                    var miniColumn = randomMiniColumns[randomMiniColumns_Index];

                    //for (int mi = 0; mi < miniColumn.CortexMemories.Count; mi += 1)
                    if (miniColumn.CortexMemories.Count > 0)
                    {
                        int mi = random.Next(miniColumn.CortexMemories.Count);
                        Memory? cortexMemory = miniColumn.CortexMemories[mi];
                        if (cortexMemory is null)
                            continue;

                        miniColumn.CortexMemories[mi] = null;

                        MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn(cortexMemory, random, cancellationToken, candidateMiniColumns);
                        if (bestForMemoryMiniColumn is not null && !ReferenceEquals(bestForMemoryMiniColumn, miniColumn))
                        {
                            bestForMemoryMiniColumn.CortexMemories.Add(cortexMemory);
                            changedCount += 1;
                        }
                        else
                        {
                            miniColumn.CortexMemories[mi] = cortexMemory;
                        }
                    }
                }                           
            }

            for (int mci = 0; mci < candidateMiniColumns.Count; mci += 1)
            {
                MiniColumn mc = candidateMiniColumns[mci];

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

            LoggersSet.UserFriendlyLogger.LogInformation($"Epoch: {epoch}/{epochCount};");

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

            if (changedCount < 1 || min_ChangesCount_UnchangedCount > 20)
                break;
        }
        

        LoggersSet.UserFriendlyLogger.LogInformation($"ReorderMemories Finished.");
    }    

    private float GetSimilarity(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];        

        float x1 = inpitItem1.Magnitude * MathF.Cos(inpitItem1.Angle);
        float y1 = inpitItem1.Magnitude * MathF.Sin(inpitItem1.Angle);
        float x2 = inpitItem2.Magnitude * MathF.Cos(inpitItem2.Angle);
        float y2 = inpitItem2.Magnitude * MathF.Sin(inpitItem2.Angle);

        var r2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
        float similarity = MathF.Exp(-r2 / 8.0f); // sigma == 2.0f

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

    public class ModelConstants : IMiniColumnsActivityConstants
    {
        /// <summary>
        ///     Радиус зоны коры в миниколонках.
        /// </summary>
        public int CortexRadius_MiniColumns => 10;

        /// <summary>
        ///     Уровень подобия для нулевой активности
        /// </summary>
        public float K0 { get; set; } = 0.66f;

        /// <summary>
        ///     Уровень подобия с пустой миниколонкой
        /// </summary>
        public float K2 { get; set; } = 1.0f; // Или чуть меньше, чем с точно таким же воспоминанием.

        /// <summary>
        ///     Порог суперактивности
        /// </summary>
        public float K4 { get; set; } = 1.0f;

        public float[] PositiveK { get; set; } = [1.00f, 0.117f, 0.050f, 0.015f];

        public float[] NegativeK { get; set; } = [1.00f, 0.117f, 0.083f, 0.020f];

        /// <summary>
        ///     Включен ли порог на суперактивность при накоплении воспоминаний
        /// </summary>
        public bool SuperactivityThreshold { get; set; } = false;
    }
}

//public double GetEnergy()
//    {
//        if (Cortex.MiniColumns is null || Cortex.InputItems.Count == 0)
//            return Double.NaN;

//        double energy = 0.0;
//        //for (int miniColumns_Index = 0; miniColumns_Index < Cortex.MiniColumns.Count; miniColumns_Index += 1)
//        //{
//        //    MiniColumn miniColumn = Cortex.MiniColumns[miniColumns_Index];            
//        //    energy += GetEnergy(miniColumn);
//        //}
//        return energy;
//    }

//public (double Average, double Minimum, double Maximum) GetAverageSimilarity()
//    {
//        if (Cortex.MiniColumns is null || Cortex.InputItems.Count == 0)
//            return (Double.NaN, Double.NaN, Double.NaN);

//        var miniColumns = Cortex.MiniColumns;

//        double similarityTotal = 0.0;
//        double similarityMin = Double.MaxValue;
//        double similarityMax = Double.MinValue;
//        for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)
//        {
//            var miniColumn = miniColumns[miniColumns_Index];

//            double similaritySubTotal = 0.0;
//            for (int i = 0; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
//            {
//                MiniColumn nearestMiniColumn = miniColumn.Temp_K_ForNearestMiniColumns[i].MiniColumn;
//                int cortexMemoriesCount = 0;
//                for (int mi = 0; mi < nearestMiniColumn.CortexMemories.Count; mi += 1)
//                {
//                    var cortexMemory = nearestMiniColumn.CortexMemories[mi];
//                    if (cortexMemory is null)
//                        continue;
//                    similaritySubTotal += GetSimilarity(miniColumn.CortexMemories[0]!, nearestMiniColumn.CortexMemories[0]!);
//                    cortexMemoriesCount += 1;
//                }
//            }
//            double similarity = similaritySubTotal / miniColumn.Temp_K_ForNearestMiniColumns.Count;
//            if (similarity < similarityMin)
//                similarityMin = similarity;
//            if (similarity > similarityMax)
//                similarityMax = similarity;
//            similarityTotal += similarity;
//            miniColumn.Temp_Distance = similarity;
//        }

//        return (Average: similarityTotal / miniColumns.Count, Minimum: similarityMin, Maximum: similarityMax);
//    }

//private double GetEnergy(MiniColumn miniColumn, Memory cortexMemory)
//    {
//        double energy = 0.0;
//        int cortexMemoriesCount = 0;
//        for (int i = 0; i < miniColumn.Temp_NearestForEnergyMiniColumns.Count; i += 1)
//        {
//            var nearestMiniColumn = miniColumn.Temp_NearestForEnergyMiniColumns[i].Item2;
//            for (int cortexMemoryIndex = 0; cortexMemoryIndex < nearestMiniColumn.CortexMemories.Count; cortexMemoryIndex += 1)
//            {
//                Memory? nearestCortexMemory = nearestMiniColumn.CortexMemories[cortexMemoryIndex];
//                if (nearestCortexMemory is not null)
//                {
//                    energy += GetEnergy(nearestCortexMemory, cortexMemory);
//                    cortexMemoriesCount += 1;
//                }
//            }
//        }
//        if (cortexMemoriesCount > 0)
//            energy /= cortexMemoriesCount;
//        return energy;
//    }


//    private double GetEnergy(Memory memory1, Memory memory2)
//    {
//        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
//        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

//        double x1 = inpitItem1.Magnitude * Math.Cos(inpitItem1.Angle);
//        double y1 = inpitItem1.Magnitude * Math.Sin(inpitItem1.Angle);
//        double x2 = inpitItem2.Magnitude * Math.Cos(inpitItem2.Angle);
//        double y2 = inpitItem2.Magnitude * Math.Sin(inpitItem2.Angle);

//        var r2 = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
//        return r2;
//    }

//    private double GetDistance(Memory memory1, Memory memory2)
//    {
//        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
//        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

//        double x1 = inpitItem1.Magnitude * Math.Cos(inpitItem1.Angle);
//        double y1 = inpitItem1.Magnitude * Math.Sin(inpitItem1.Angle);
//        double x2 = inpitItem2.Magnitude * Math.Cos(inpitItem2.Angle);
//        double y2 = inpitItem2.Magnitude * Math.Sin(inpitItem2.Angle);

//        var r2 = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
//        return Math.Sqrt(r2);
//    }