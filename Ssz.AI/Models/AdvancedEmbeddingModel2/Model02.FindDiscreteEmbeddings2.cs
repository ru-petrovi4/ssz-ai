//using System;
//using System.Collections.Generic;
//using System.Collections.ObjectModel;
//using System.Diagnostics;
//using System.Drawing;
//using System.Globalization;
//using System.IO;
//using System.IO.Compression;
//using System.Linq;
//using System.Numerics.Tensors;
//using System.Text.Json;
//using System.Threading;
//using System.Threading.Tasks;
//using MathNet.Numerics;
//using MathNet.Numerics.Providers.LinearAlgebra;
//using Microsoft.Extensions.Configuration;
//using Microsoft.Extensions.Hosting;
//using Microsoft.Extensions.Logging;
//using Microsoft.Extensions.Logging.Abstractions;
//using Ssz.AI.Core;
//using Ssz.AI.Helpers;
//using Ssz.AI.ViewModels;
//using Ssz.Utils;
//using Ssz.Utils.Addons;
//using Ssz.Utils.Logging;
//using Ssz.Utils.Serialization;

//namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

//public class Model02
//{
//    public const string FileName_Cortex02 = "AdvancedEmbedding2_Cortex02.bin";

//    #region construction and destruction

//    public Model02()
//    {
//        LoggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
//    }

//    #endregion

//    #region public functions       

//    public ILoggersSet LoggersSet { get; }

//    public static readonly ModelConstants Constants = new();

//    public InputCorpusData InputCorpusData = null!;    

//    public Cortex Cortex = null!;

//    public void PrepareCalculate(Random random)
//    {
//        InputCorpusData = InputCorpusDataHelper.GetInputCorpusData(random, Constants.DiscreteVectorLength);

//        Cortex = new Cortex(Constants);
//        Cortex.GenerateOwnedData(InputCorpusData.Words);
//        Cortex.Prepare();
//    }

//    public bool CalculateCortexMemories(int cortexMemoriesCount, Random random)
//    {
//        return Cortex.CalculateCortexMemories(InputCorpusData, cortexMemoriesCount, random, LoggersSet.UserFriendlyLogger);
//    }

//    public async Task ReorderMemoriesAsync(int epochCount, Random random, Func<Task>? epochRefreshAction = null)
//    {
//        await Cortex.ReorderMemoriesAsync(epochCount, random, LoggersSet.UserFriendlyLogger, epochRefreshAction);
//    }        

//    public VisualizationWithDesc[] GetImageWithDescs()
//    {
//        var bitmapFromMiniColums_ActivityColor = Visualisation.GetBitmapFromMiniColums_ActivityColor(Cortex);
//        var bitmapFromMiniColums_SuperActivityColor = Visualisation.GetBitmapFromMiniColums_SuperActivityColor(Cortex, null);
        
//        var bitmapFromMiniColums_ActivityColor_WordCode = Visualisation.GetBitmapFromMiniColums_ActivityColor_Code(Cortex);
//        var bitmapFromMiniColums_SuperActivityColor_WordCode = Visualisation.GetBitmapFromMiniColums_SuperActivityColor_Code(Cortex);

//        return [                      
//                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_ActivityColor),
//                    Desc = @"Активность миниколонок (белый - максимум)" },
//                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_ActivityColor_WordCode),
//                    Desc = @"Активность миниколонок, код слова" },
//                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_SuperActivityColor),
//                    Desc = @"Суперактивность миниколонок (белый - максимум)" },
//                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_SuperActivityColor_WordCode),
//                    Desc = @"Суперактивность миниколонок, код слова" },
//                new Model3DWithDesc { Data = Visualization3D.Get_MiniColumnsMemories_Model3DScene(Cortex),
//                    Desc = $"Накопленные воспоминания в миниколонках." },
//                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex)),
//                    Desc = @"Средний цвет накопленных воспоминаний в миниколонках" },
//                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex)),
//                    Desc = @"Количество воспоминаний в миниколонках" }
//            ];
//    }

//    #endregion

//    #region private functions    



//    #endregion

//    #region private fields    

//    #endregion

//    public class ModelConstants : IMiniColumnsActivityConstants
//    {
//        public int DiscreteVectorLength => 300;

//        /// <summary>
//        ///     Количество миниколонок в зоне коры по оси X
//        /// </summary>
//        public int CortexWidth_MiniColumns => 17;

//        /// <summary>
//        ///     Количество миниколонок в зоне коры по оси Y
//        /// </summary>
//        public int CortexHeight_MiniColumns => 17;

//        public double SuperActivityRadius_MiniColumns => 3;

//        /// <summary>
//        ///     Нулевой уровень косинусного подобия
//        /// </summary>
//        public float K0 { get; set; } = 0.11f; // 0.12

//        public float K1 { get; set; } = 0.2f;

//        /// <summary>
//        ///     Косинусное подобие с пустой миниколонкой
//        /// </summary>
//        public float K2 { get; set; } = 0.96f;

//        public float K3 { get; set; } = 0.2f;

//        /// <summary>
//        ///     Порог суперактивности
//        /// </summary>
//        public float K4 { get; set; } = 0.2f;

//        public float[] PositiveK { get; set; } = [1.00f, 0.14f, 0.025f, 0.00f];

//        public float[] NegativeK { get; set; } = [1.00f, 0.14f, 0.07f, 0.00f];

//        /// <summary>
//        ///     Включен ли порог на суперактивность при накоплении воспоминаний
//        /// </summary>
//        public bool SuperactivityThreshold { get; set; }
//    }
//}