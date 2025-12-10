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

        float maxMagnitude = Single.MinValue;
        for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)            
        {
            MiniColumn miniColumn = miniColumns[miniColumns_Index];            

            InputItem inputItem = new();
            inputItem.Index = Cortex.InputItems.Count;
            inputItem.Angle = MathHelper.NormalizeAngle(MathF.Atan2(miniColumn.MCY, miniColumn.MCX));
            inputItem.Magnitude = MathF.Sqrt(miniColumn.MCY * miniColumn.MCY + miniColumn.MCX * miniColumn.MCX);
            if (inputItem.Magnitude > maxMagnitude)
                maxMagnitude = inputItem.Magnitude;

            var cortexMemeory = new Memory
            {
                InputItemIndex = inputItem.Index
            };

            if (isRandom)
            {
                randomMiniColumns[miniColumns_Index].CortexMemories.Add(cortexMemeory);                
            }
            else
            {
                miniColumn.CortexMemories.Add(cortexMemeory);
            }   

            Cortex.InputItems.Add(inputItem);
        }

        for (int inputItem_Index = 0; inputItem_Index < Cortex.InputItems.Count; inputItem_Index += 1)
        {
            InputItem inputItem = Cortex.InputItems[inputItem_Index];
            float s = MathF.Sqrt(inputItem.Magnitude / maxMagnitude);            
            inputItem.Color = Visualisation.ColorFromHSV((double)(inputItem.Angle + MathF.PI) / (2 * MathF.PI), s, 1.0);
        }
    }

    public async Task ReorderMemoriesAsync(int epochCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        await ReorderMemoriesAsync(epochCount, random, cancellationToken, refreshAction, mc => mc.Temp_CandidateForSwapMiniColumns);
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
        for (int miniColumns_Index = 0; miniColumns_Index < Cortex.MiniColumns.Count; miniColumns_Index += 1)
        {
            MiniColumn miniColumn = Cortex.MiniColumns[miniColumns_Index];            
            energy += GetEnergy(miniColumn);
        }
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
        for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Count; miniColumns_Index += 1)
        {
            var miniColumn = miniColumns[miniColumns_Index];

            double distanceSubTotal = 0.0;
            for (int i = 0; i < miniColumn.Temp_NearestForEnergyMiniColumns.Count; i += 1)
            {
                MiniColumn candidateForSwapMiniColumn = miniColumn.Temp_NearestForEnergyMiniColumns[i].Item2;
                distanceSubTotal += GetDistance(miniColumn.CortexMemories[0]!, candidateForSwapMiniColumn.CortexMemories[0]!);
            }
            double distance = distanceSubTotal / miniColumn.Temp_NearestForEnergyMiniColumns.Count;
            if (distance < distanceMin)
                distanceMin = distance;
            if (distance > distanceMax)
                distanceMax = distance;
            distanceTotal += distance;
            miniColumn.Temp_Distance = distance;
        }

        return (Average: distanceTotal / miniColumns.Count, Minimum: distanceMin, Maximum: distanceMax);
    }

    #endregion

    #region private functions     

    private async Task ReorderMemoriesAsync(int epochCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction, Func<MiniColumn, FastList<(double, MiniColumn)>> getCandidateForSwapMiniColumns)
    {
        for (int epoch = 0; epoch < epochCount; epoch += 1)
        {
            var randomMiniColumns = Cortex.MiniColumns.ToArray();
            random.Shuffle(randomMiniColumns);

            bool changed = false;            

            for (int randomMiniColumns_Index = 0; randomMiniColumns_Index < randomMiniColumns.Length; randomMiniColumns_Index += 1)
            {
                var miniColumn = randomMiniColumns[randomMiniColumns_Index];

                var candidateForSwapMiniColumns = getCandidateForSwapMiniColumns(miniColumn);

                miniColumn.Temp_Energy = GetEnergy(miniColumn);                
                for (int i = 0; i < candidateForSwapMiniColumns.Count; i += 1)
                {
                    MiniColumn candidateForSwapMiniColumn = candidateForSwapMiniColumns[i].Item2;
                    candidateForSwapMiniColumn.Temp_Energy = GetEnergy(candidateForSwapMiniColumn);                    
                }

                double minEnergy = 0.0f;
                MiniColumn minEnergy_MiniColumn = miniColumn;

                for (int i = 0; i < candidateForSwapMiniColumns.Count; i += 1)
                {
                    MiniColumn candidateForSwapMiniColumn = candidateForSwapMiniColumns[i].Item2;

                    miniColumn.CortexMemories.Swap(candidateForSwapMiniColumn.CortexMemories);
                    double energy = -miniColumn.Temp_Energy - candidateForSwapMiniColumn.Temp_Energy
                        + GetEnergy(miniColumn) + GetEnergy(candidateForSwapMiniColumn);
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

            LoggersSet.UserFriendlyLogger.LogInformation($"Epoch: {epoch}/{epochCount};");
            await refreshAction();

            if (!changed)
                break;
        }
    }

    private double GetEnergy(MiniColumn miniColumn)
    {
        double energy = 0.0;
        for (int i = 0; i < miniColumn.Temp_NearestForEnergyMiniColumns.Count; i += 1)
        {
            var it = miniColumn.Temp_NearestForEnergyMiniColumns[i];
            energy += GetDistance(miniColumn.CortexMemories[0]!, it.Item2.CortexMemories[0]!);
        }
        return energy / miniColumn.Temp_NearestForEnergyMiniColumns.Count;
    }

    private double GetEnergy_SpringModel(MiniColumn miniColumn)
    {
        double energy = 0.0;
        for (int i = 0; i < miniColumn.Temp_NearestForEnergyMiniColumns.Count; i += 1)
        {
            var it = miniColumn.Temp_NearestForEnergyMiniColumns[i];
            energy += GetSimilarity(miniColumn.CortexMemories[0]!, it.Item2.CortexMemories[0]!) * it.Item1;
        }
        return energy;
    }

    private double GetDistance(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

        double x1 = inpitItem1.Magnitude * Math.Cos(inpitItem1.Angle);
        double y1 = inpitItem1.Magnitude * Math.Sin(inpitItem1.Angle);
        double x2 = inpitItem2.Magnitude * Math.Cos(inpitItem2.Angle);
        double y2 = inpitItem2.Magnitude * Math.Sin(inpitItem2.Angle);

        var d = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        return Math.Sqrt(d);
    }

    private double GetSimilarity(Memory memory1, Memory memory2)
    {
        InputItem inpitItem1 = Cortex.InputItems[memory1.InputItemIndex];
        InputItem inpitItem2 = Cortex.InputItems[memory2.InputItemIndex];

        double x1 = inpitItem1.Magnitude * Math.Cos(inpitItem1.Angle);
        double y1 = inpitItem1.Magnitude * Math.Sin(inpitItem1.Angle);
        double x2 = inpitItem2.Magnitude * Math.Cos(inpitItem2.Angle);
        double y2 = inpitItem2.Magnitude * Math.Sin(inpitItem2.Angle);

        var d = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));            
        return - Math.Pow(d, 0.5);
    }

    #endregion

    #region private fields    

    #endregion

    public class ModelConstants : IModelConstants
    {
        /// <summary>
        ///     Радиус зоны коры в миниколонках.
        /// </summary>
        public int CortexRadius_MiniColumns => 10;
    }
}