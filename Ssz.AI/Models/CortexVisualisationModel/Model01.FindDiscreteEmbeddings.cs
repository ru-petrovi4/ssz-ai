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

public class Model01
{
    public const string FileName_Cortex = "CortexVisualisationModel_Cortex.bin";

    #region construction and destruction

    public Model01()
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
        return [
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex)),
                    Desc = @"Воспоминания в миниколонках" }
            ];
    }

    public void PutInitialMemoriesPinwheel(Random random, bool isRandom)
    {
        if (Cortex.MiniColumns is null)
            return;

        int center_MCX = Cortex.MiniColumns.Dimensions[0] / 2;
        int center_MCY = Cortex.MiniColumns.Dimensions[1] / 2;
        float maxRadius = MathF.Sqrt(center_MCX * center_MCX + center_MCY * center_MCY);

        var miniColumns = Cortex.MiniColumns.Data.OfType<MiniColumn>().ToArray();
        var randomMiniColumns = (MiniColumn[])miniColumns.Clone();
        random.Shuffle(randomMiniColumns);        

        for (int miniColumns_Index = 0; miniColumns_Index < miniColumns.Length; miniColumns_Index += 1)            
        {
            MiniColumn? miniColumn = miniColumns[miniColumns_Index];
            if (miniColumn is null)
                continue;

            InputItem inputItem = new();
            inputItem.Index = Cortex.InputItems.Count;
            inputItem.Angle = MathHelper.NormalizeAngle(MathF.Atan2(miniColumn.MCY - center_MCY, miniColumn.MCX - center_MCX));
            inputItem.Magnitude = MathF.Sqrt((miniColumn.MCY - center_MCY) * (miniColumn.MCY - center_MCY) + (miniColumn.MCX - center_MCX) * (miniColumn.MCX - center_MCX)) / maxRadius;
            inputItem.Color = Visualisation.ColorFromHSV((double)(inputItem.Angle + MathF.PI) / (2 * MathF.PI), inputItem.Magnitude, 1.0);

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
    }

    public async Task ReorderMemoriesAsync(int epochCount, Random random, CancellationToken cancellationToken, Func<Task> refreshAction)
    {
        for (int epoch = 0; epoch < epochCount; epoch += 1)
        {
            var randomMiniColumns = Cortex.MiniColumns.Data.OfType<MiniColumn>().ToArray();
            random.Shuffle(randomMiniColumns);

            bool changed = false;

            for (int randomMiniColumns_Index = 0; randomMiniColumns_Index < randomMiniColumns.Length; randomMiniColumns_Index += 1)
            {
                var miniColumn = randomMiniColumns[randomMiniColumns_Index];

                miniColumn.Temp_Energy = GetEnergy(miniColumn);
                double initialEnergy = miniColumn.Temp_Energy;                
                for (int i = 0; i < miniColumn.Temp_AdjacentMiniColumns.Count; i += 1)
                {
                    MiniColumn adjacentMiniColumn = miniColumn.Temp_AdjacentMiniColumns[i];
                    adjacentMiniColumn.Temp_Energy = GetEnergy(adjacentMiniColumn);
                    initialEnergy += adjacentMiniColumn.Temp_Energy;
                }

                double minEnergy = 0.0f;
                MiniColumn minEnergy_MiniColumn = miniColumn;

                for (int i = 0; i < miniColumn.Temp_AdjacentMiniColumns.Count; i += 1)
                {
                    MiniColumn adjacentMiniColumn = miniColumn.Temp_AdjacentMiniColumns[i];

                    miniColumn.CortexMemories.Swap(adjacentMiniColumn.CortexMemories);
                    double energy = - miniColumn.Temp_Energy - adjacentMiniColumn.Temp_Energy
                        + GetEnergy(miniColumn) + GetEnergy(adjacentMiniColumn);
                    miniColumn.CortexMemories.Swap(adjacentMiniColumn.CortexMemories);

                    if (energy < minEnergy)
                    {
                        minEnergy = energy;
                        minEnergy_MiniColumn = adjacentMiniColumn;
                    }
                }

                if (minEnergy_MiniColumn != miniColumn)
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

    #endregion

    #region private functions    

    private double GetEnergy(MiniColumn miniColumn)
    {
        double energy = 0.0;
        for (int i = 0; i < miniColumn.Temp_K_ForNearestMiniColumns.Count; i += 1)
        {
            var it = miniColumn.Temp_K_ForNearestMiniColumns[i];
            energy += GetSimilarity(miniColumn.CortexMemories[0]!, it.Item2.CortexMemories[0]!) * it.Item1;
        }
        return energy;
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
        return -d;
    }

    #endregion

    #region private fields    

    #endregion

    public class ModelConstants
    {        
        /// <summary>
        ///     Количество миниколонок в зоне коры по оси X
        /// </summary>
        public int CortexWidth_MiniColumns => 17;

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси Y
        /// </summary>
        public int CortexHeight_MiniColumns => 17;        
    }
}