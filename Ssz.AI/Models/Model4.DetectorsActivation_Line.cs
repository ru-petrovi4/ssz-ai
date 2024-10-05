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

            //List<GradientInPoint[,]> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(images[i], MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);
                //gradientMatricesCollection.Add(gradientMatrix);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            _detectors = DetectorsGenerator.Generate(gradientDistribution);
        }

        #endregion

        #region public functions

        public const int AngleRangesCount = 4;

        public const int MagnitudeRangesCount = 4;

        public Image[] GetImages(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280
            int width = 280;
            int height = 280;

            int centerX = (int)(width / 2.0 + positionK * width);
            int centerY = (int)(height / 2.0);

            double angle = Math.PI / 2 + angleK * 2 * Math.PI;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(centerX + lineLength * Math.Cos(angle));
            int endY = (int)(centerY + lineLength * Math.Sin(angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(centerX - lineLength * Math.Cos(angle));
            int startY = (int)(centerY - lineLength * Math.Sin(angle));

            Bitmap originalBitmap = new Bitmap(width, height);
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
            Bitmap resizedBitmap = new Bitmap(MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);
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
                g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.ImageWidth, MNISTHelper.ImageHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);

            List<Detector> activatedDetectors = _detectors.Where(d => d.IsActivated(gradientMatrix)).ToList();

            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
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

        #endregion        
    }
}
