using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.DrawingCore.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model5
    {
        #region construction and destruction

        /// <summary>
        ///     Построение "вертушки"
        /// </summary>
        public Model5()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            List<GradientInPoint[,]> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                gradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            _detectors = DetectorsGenerator.Generate(gradientDistribution, Constants.AngleRangesCount, Constants.MagnitudeRangesCount, Constants.HashLength);

            Cortex = new Cortex(Constants, _detectors);

            // Прогон всех картинок            
            foreach (var gradientMatrix in gradientMatricesCollection)
            {                
                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    i =>
                {
                    var d = Cortex.SubArea_Detectors[i];
                    d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                });

                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_MiniColumns.Length,
                    i =>
                    {
                        var mc = Cortex.SubArea_MiniColumns[i];
                        mc.Temp_Activity = mc.GetActivity();
                        //subAreaMiniColums[i] = mc;
                    });

                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_MiniColumns.Length,
                    i =>
                    {
                        var mc = Cortex.SubArea_MiniColumns[i];
                        mc.Temp_SuperActivity = mc.GetSuperActivity();
                        //subAreaMiniColums[i] = mc;
                    });
            }
        }

        #endregion

        #region public functions

        public readonly ModelConstants Constants = new();        

        public Cortex Cortex { get; }        

        #endregion        

        #region private fields

        private List<Detector> _detectors;
        /// <summary>
        ///     Начальная картина активации детекторов (до смещения).
        /// </summary>
        public Bitmap _detectorsActivationBitmap0 = null!;

        #endregion

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : ICortexConstants
        {
            /// <summary>
            ///     Ширина основного изображения
            /// </summary>
            public int ImageWidth => MNISTHelper.MNISTImageWidth;

            /// <summary>
            ///     Высота основного изображения
            /// </summary>
            public int ImageHeight => MNISTHelper.MNISTImageHeight;

            public int AngleRangesCount => 4;

            public int MagnitudeRangesCount => 4;

            public int GeneratedImageWidth => 280;

            public int GeneratedImageHeight => 280;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси X
            /// </summary>
            public int CortexWidth => 200;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси Y
            /// </summary>
            public int CortexHeight => 200;

            /// <summary>
            ///     Площадь одного детектрора   
            /// </summary>
            public double DetectorArea => 0.01;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;            

            public int HashLength => 200;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 400;

            /// <summary>
            ///     Индекс X центра подобласти
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти
            /// </summary>
            public int SubAreaCenter_Cy => 100;           

            /// <summary>
            ///     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
            /// </summary>
            public int InitialMemoryBitsCount => 11;

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 7;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int NearestMiniColumnsDelta => 5;

            public double NearestMiniColumnsK => 5;
        }
    }
}
