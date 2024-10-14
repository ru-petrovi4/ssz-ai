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
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model4
    {
        #region construction and destruction

        /// <summary>
        ///     Построение графика распределения венлечин градиентов
        /// </summary>
        public Model4()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new()
            {
                MagnitudeData = new UInt64[SobelOperator.MagnitudeUpperLimit],
                AngleData = new UInt64[360]
            };

            List<GradientInPoint[,]> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                gradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            _detectors = DetectorsGenerator.Generate(gradientDistribution);

            // Вызываем для вычисления начального вектора активации детекторов
            GetImages(0.0, 0.0);

            Cortex = new(
                CortexWidth,
                CortexHeight,
                MNISTHelper.MNISTImageWidth,
                MNISTHelper.MNISTImageHeight,
                MiniColumnVisibleDetectorsCount,
                0.01,
                HashLength,
                _detectors
                );

            DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            foreach (var gradientMatrix in gradientMatricesCollection)
            {
                List<Detector> activatedDetectors = _detectors.Where(d => d.IsActivated(gradientMatrix)).ToList();

                foreach (var miniColumn in Cortex.MiniColumns)
                {                    
                    dataToDisplayHolder.MiniColumsActiveBitsDistribution[activatedDetectors.Intersect(miniColumn.Detectors).Count()] += 1;
                }                
            }
        }

        #endregion

        #region public functions

        public const int AngleRangesCount = 4;

        public const int MagnitudeRangesCount = 4;

        public const int GeneratedImageWidth = 280;
        public const int GeneratedImageHeight = 280;

        public const int CortexWidth = 200;
        public const int CortexHeight = 200;

        public const int MiniColumnVisibleDetectorsCount = 250;

        public const int HashLength = 200;

        public double DetectorsActivationScalarProduct0 { get; set; }
        public double DetectorsActivationScalarProduct { get; set; }

        public int CenterX { get; set; }
        public int CenterXDelta { get; set; }
        public int CenterY { get; set; }
        public double AngleDelta { get; set; }
        public double Angle { get; set; }

        public Cortex Cortex { get; }

        public Image[] GetImages(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            CenterXDelta = (int)(positionK * GeneratedImageWidth / 2.0); 
            CenterX = (int)(GeneratedImageWidth / 2.0) + CenterXDelta;
            CenterY = (int)(GeneratedImageHeight / 2.0);

            AngleDelta = angleK * 2.0 * Math.PI;
            Angle = Math.PI / 2 + AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(CenterX + lineLength * Math.Cos(Angle));
            int endY = (int)(CenterY + lineLength * Math.Sin(Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(CenterX - lineLength * Math.Cos(Angle));
            int startY = (int)(CenterY - lineLength * Math.Sin(Angle));

            Bitmap originalBitmap = new Bitmap(GeneratedImageWidth, GeneratedImageHeight);
            using (Graphics g = Graphics.FromImage(originalBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Создаем кисть и устанавливаем толщину линии
                using (Pen pen = new Pen(Color.White, 15))
                {                    
                    // Рисуем наклонную линию
                    g.DrawLine(pen, startX, startY, endX, endY);
                }
            }

            // Уменьшаем изображение до размера 28x28

            // Создаем пустое изображение 28x28
            Bitmap resizedBitmap = new Bitmap(MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
            using (Graphics g = Graphics.FromImage(resizedBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры для уменьшения
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Масштабируем изображение
                g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            List<Detector> activatedDetectors = _detectors.Where(d => d.IsActivated(gradientMatrix)).ToList();

            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);
            if (positionK == 0.0 && angleK == 0.0)
            {
                _detectorsActivationBitmap0 = detectorsActivationBitmap;
            }

            double detectorsActivationScalarProduct = 0.0;
            for (int y = 0; y < _detectorsActivationBitmap0.Height; y += 1)
            {
                for (int x = 0; x < _detectorsActivationBitmap0.Width; x += 1)
                {
                    var p0 = _detectorsActivationBitmap0.GetPixel(x, y);
                    var p = detectorsActivationBitmap.GetPixel(x, y);
                    if (p0.R > 0 && p.R > 0)
                        detectorsActivationScalarProduct += 1.0;
                }
            }
            DetectorsActivationScalarProduct = detectorsActivationScalarProduct;

            if (positionK == 0.0 && angleK == 0.0)
            {
                DetectorsActivationScalarProduct0 = detectorsActivationScalarProduct;
            }
            
            return [originalBitmap, resizedBitmap, gradientBitmap, detectorsActivationBitmap];
        }

        #endregion

        #region private functions

        private UInt64[] GetAccumulativeDistribution(UInt64[] distribution)
        {
            UInt64[] result = new UInt64[distribution.Length];
            UInt64 value = 0;
            foreach (int i in Enumerable.Range(0, distribution.Length))
            {
                value += distribution[i];
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        ///     Returns highLimitIndex > lowLimitIndex
        /// </summary>
        /// <param name="accumulativeDistribution"></param>
        /// <param name="random"></param>
        /// <param name="rangesCount"></param>
        /// <returns></returns>
        private (int lowLimitIndex, int highLimitIndex) GetLimitsIndices(UInt64[] accumulativeDistribution, Random random, int rangesCount)
        {
            UInt64 maxSamples = accumulativeDistribution[^1]; // Последний элемент массиваж
            UInt64 rangeSamples = (maxSamples / (UInt64)rangesCount);
            UInt64 lowLimitSamples = (UInt64)(random.NextDouble() * maxSamples);
            UInt64 hightLimitSamples = lowLimitSamples + rangeSamples;
            int lowLimitIndex = 0;
            foreach (int i in Enumerable.Range(0, accumulativeDistribution.Length))
            {
                if (lowLimitSamples < accumulativeDistribution[i])
                {
                    lowLimitIndex = i;
                    break;
                }
            }
            int highLimitIndex = accumulativeDistribution.Length;
            foreach (int i in Enumerable.Range(lowLimitIndex + 1, accumulativeDistribution.Length - lowLimitIndex - 1))
            {
                if (hightLimitSamples < accumulativeDistribution[i])
                {
                    highLimitIndex = i;
                    break;
                }
            }
            return (lowLimitIndex, highLimitIndex);
        }

        #endregion

        #region private fields

        private List<Detector> _detectors;
        /// <summary>
        ///     Начальная картина активации детекторов (до смещения).
        /// </summary>
        public Bitmap _detectorsActivationBitmap0 = null!;

        #endregion        
    }
}
