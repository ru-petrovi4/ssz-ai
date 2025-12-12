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

    public VisualizationWithDesc[] GetImageWithDescs()
    {
        var it = GetAverageDistance();
        return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex)),
                    Desc = $"Воспоминания в миниколонках.\nЭнергия: {GetEnergy()}" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsValue(Cortex, (MiniColumn mc) => mc.Temp_Distance, valueMin: 0.0, valueMax: 15.0)),
                    Desc = $"Среднее расстояние: {it.Average}\nМинимальное: {it.Minimum}\nМаксимальное: {it.Maximum}" }
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

            InputItem inputItem = AddInputItem(random, miniColumn);                  

            var cortexMemory = new Memory
            {
                InputItemIndex = inputItem.Index
            };

            if (isRandom)
            {
                randomMiniColumns[miniColumns_Index].CortexMemories.Add(cortexMemory);                
            }
            else
            {
                miniColumn.CortexMemories.Add(cortexMemory);
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

    public async Task ReorderMemoriesAsync(int epochCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        await ReorderMemoriesAsync(epochCount, random, cancellationToken, refreshAction, Cortex.MiniColumns);
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

    public double GetEnergy()
    {
        if (Cortex.MiniColumns is null || Cortex.InputItems.Count == 0)
            return Double.NaN;

        double energy = 0.0;
        //for (int miniColumns_Index = 0; miniColumns_Index < Cortex.MiniColumns.Count; miniColumns_Index += 1)
        //{
        //    MiniColumn miniColumn = Cortex.MiniColumns[miniColumns_Index];            
        //    energy += GetEnergy(miniColumn);
        //}
        return energy;
    }

    public (double Average, double Minimum, double Maximum) GetAverageDistance()
    {
        if (Cortex.MiniColumns is null || Cortex.InputItems.Count == 0)
            return (Double.NaN, Double.NaN, Double.NaN);

        var miniColumns = Cortex.MiniColumns;        

        double distanceTotal = 0.0;
        double distanceMin = Double.MaxValue;
        double distanceMax = Double.MinValue;
        //for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)
        //{
        //    var miniColumn = miniColumns[miniColumns_Index];

        //    double distanceSubTotal = 0.0;
        //    for (int i = 0; i < miniColumn.Temp_NearestForEnergyMiniColumns.Count; i += 1)
        //    {
        //        MiniColumn candidateForSwapMiniColumn = miniColumn.Temp_NearestForEnergyMiniColumns[i].Item2;
        //        distanceSubTotal += GetDistance(miniColumn.CortexMemories[0]!, candidateForSwapMiniColumn.CortexMemories[0]!);
        //    }
        //    double distance = distanceSubTotal / miniColumn.Temp_NearestForEnergyMiniColumns.Count;
        //    if (distance < distanceMin)
        //        distanceMin = distance;
        //    if (distance > distanceMax)
        //        distanceMax = distance;
        //    distanceTotal += distance;
        //    miniColumn.Temp_Distance = distance;
        //}

        return (Average: distanceTotal / miniColumns.Count, Minimum: distanceMin, Maximum: distanceMax);
    }

    #endregion

    #region private functions   

    private InputItem AddInputItem(Random random, MiniColumn miniColumn)
    {
        InputItem inputItem = new();
        inputItem.Index = Cortex.InputItems.Count;
        inputItem.Angle = MathHelper.NormalizeAngle(MathF.Atan2(miniColumn.MCY, miniColumn.MCX));
        inputItem.Magnitude = MathF.Sqrt(miniColumn.MCY * miniColumn.MCY + miniColumn.MCX * miniColumn.MCX);

        float s = MathF.Sqrt(inputItem.Magnitude / (Constants.CortexRadius_MiniColumns + 1));
        inputItem.Color = Visualisation.ColorFromHSV((double)(inputItem.Angle + MathF.PI) / (2 * MathF.PI), s, 1.0);

        Cortex.InputItems.Add(inputItem);
        return inputItem;
    }

    private Memory CreateMemory(Random random)
    {
        MiniColumn miniColumn = Cortex.MiniColumns[random.Next(Cortex.MiniColumns.Count)];

        InputItem inputItem = AddInputItem(random, miniColumn);

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

    private async Task ReorderMemoriesAsync(int epochCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction, FastList<MiniColumn> candidateMiniColumns)
    {
        int minChangesCount = Int32.MaxValue;
        int minChangesCount_UnchangedCount = 0;

        for (int epoch = 0; epoch < epochCount; epoch += 1)
        {
            var randomMiniColumns = Cortex.MiniColumns.ToArray();
            random.Shuffle(randomMiniColumns);

            int changedCount = 0;

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

                    MiniColumn? bestForMemoryMiniColumn = FindBestForMemoryMiniColumn(cortexMemory, random, cancellationToken, Cortex.MiniColumns);
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

            LoggersSet.UserFriendlyLogger.LogInformation($"Epoch: {epoch}/{epochCount};");

            if (refreshAction is not null)
                await refreshAction();

            if (changedCount < minChangesCount)
            {
                minChangesCount_UnchangedCount = 0;
                minChangesCount = changedCount;
            }
            else
            {
                minChangesCount_UnchangedCount += 1;
            }

            if (changedCount < 1 || minChangesCount_UnchangedCount > 20)
                break;
        }

        LoggersSet.UserFriendlyLogger.LogInformation($"ReorderMemories Finished.");
    }

    private double GetEnergy(MiniColumn miniColumn, Memory cortexMemory)
    {
        double energy = 0.0;
        int cortexMemoriesCount = 0;
        for (int i = 0; i < miniColumn.Temp_NearestForEnergyMiniColumns.Count; i += 1)
        {
            var nearestMiniColumn = miniColumn.Temp_NearestForEnergyMiniColumns[i].Item2;
            for (int cortexMemoryIndex = 0; cortexMemoryIndex < nearestMiniColumn.CortexMemories.Count; cortexMemoryIndex += 1)
            {
                Memory? nearestCortexMemory = nearestMiniColumn.CortexMemories[cortexMemoryIndex];
                if (nearestCortexMemory is not null)
                {
                    energy += GetEnergy(nearestCortexMemory, cortexMemory);
                    cortexMemoriesCount += 1;
                }
            }                
        }
        if (cortexMemoriesCount > 0)
            energy /= cortexMemoriesCount;
        return energy;
    }


    private double GetEnergy(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

        double x1 = inpitItem1.Magnitude * Math.Cos(inpitItem1.Angle);
        double y1 = inpitItem1.Magnitude * Math.Sin(inpitItem1.Angle);
        double x2 = inpitItem2.Magnitude * Math.Cos(inpitItem2.Angle);
        double y2 = inpitItem2.Magnitude * Math.Sin(inpitItem2.Angle);

        var r2 = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        return r2;
    }

    private double GetDistance(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

        double x1 = inpitItem1.Magnitude * Math.Cos(inpitItem1.Angle);
        double y1 = inpitItem1.Magnitude * Math.Sin(inpitItem1.Angle);
        double x2 = inpitItem2.Magnitude * Math.Cos(inpitItem2.Angle);
        double y2 = inpitItem2.Magnitude * Math.Sin(inpitItem2.Angle);

        var r2 = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        return Math.Sqrt(r2);
    }

    private float GetSimilarity(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

        float x1 = inpitItem1.Magnitude * MathF.Cos(inpitItem1.Angle);
        float y1 = inpitItem1.Magnitude * MathF.Sin(inpitItem1.Angle);
        float x2 = inpitItem2.Magnitude * MathF.Cos(inpitItem2.Angle);
        float y2 = inpitItem2.Magnitude * MathF.Sin(inpitItem2.Angle);

        var r = MathF.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));            
        return MathHelper.NormalPdfF(r, 0.0f, 3.0f);
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
        public float K0 { get; set; } = 0.13f;

        /// <summary>
        ///     Уровень подобия с пустой миниколонкой
        /// </summary>
        public float K2 { get; set; } = 0.13f;

        /// <summary>
        ///     Порог суперактивности
        /// </summary>
        public float K4 { get; set; } = 0.13f;

        public float[] PositiveK { get; set; } = [1.00f, 0.14f, 0.025f];

        public float[] NegativeK { get; set; } = [1.00f, 0.14f, 0.07f];

        /// <summary>
        ///     Включен ли порог на суперактивность при накоплении воспоминаний
        /// </summary>
        public bool SuperactivityThreshold { get; set; } = false;
    }
}