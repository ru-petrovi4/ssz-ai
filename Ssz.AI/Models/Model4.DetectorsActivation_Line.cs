using Avalonia.Controls;
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
        public const int AngleRangesCount = 4;

        public const int MagnitudeRangesCount = 4;        

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

            List<Detector> detectors = DetectorsGenerator.Generate(gradientDistribution);            

            // Создаем изображение размером 280x280
            int width = 280;
            int height = 280;
            Bitmap bitmap = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем качество сглаживания
                g.SmoothingMode = SmoothingMode.AntiAlias;

                // Создаем кисть и устанавливаем толщину линии
                using (Pen pen = new Pen(Color.White, 15))
                {
                    // Рисуем наклонную линию
                    g.DrawLine(pen, 0, 0, width, height);
                }
            }

            // Уменьшаем изображение до размера 28x28
            Bitmap resizedBitmap = new Bitmap(bitmap, new Size(MNISTHelper.ImageWidth, MNISTHelper.ImageHeight));
            // Применяем оператор Собеля к первому изображению
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);

            List<Detector> activatedDetectors = detectors.Where(d => d.IsActivated(gradientMatrix)).ToList();            
                     
            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);            
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            VisualisationHelper.ShowImages([bitmap, resizedBitmap, gradientBitmap, detectorsActivationBitmap]);
        }

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
    }
}
