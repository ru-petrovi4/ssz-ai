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

        var randomMiniColumns = Cortex.MiniColumns.Data.ToArray();        
        random.Shuffle(randomMiniColumns);        

        for (int randomMiniColumns_Index = 0; randomMiniColumns_Index < Cortex.MiniColumns.Data.Length; randomMiniColumns_Index += 1)            
        {
            MiniColumn miniColumn = Cortex.MiniColumns.Data[randomMiniColumns_Index];

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
                randomMiniColumns[randomMiniColumns_Index].CortexMemories.Add(cortexMemeory);                
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
        await Task.Delay(0);
    }

    #endregion

    #region private functions    



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