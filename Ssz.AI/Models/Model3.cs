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
using System.Linq;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model3
    {        
        public const int AngleRangesCount = 4;

        public const int MagnitudeRangesCount = 4;        

        /// <summary>
        ///     Построение графика распределения венлечин градиентов
        /// </summary>
        public Model3()
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
                GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(images[i], MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);
                //gradientMatricesCollection.Add(gradientMatrix);
                SobelOperator.CalculateDistribution(gradientMatrix, gradientDistribution);
            }

            UInt64[] magnitudeAccumulativeDistribution = GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
            UInt64[] angleAccumulativeDistribution = GetAccumulativeDistribution(gradientDistribution.AngleData);                  

            Random random = new();
            List<Detector> detectors = new(MNISTHelper.ImageWidth * MNISTHelper.ImageHeight * 100);
            foreach (int i in Enumerable.Range(0, (MNISTHelper.ImageWidth - 1) * 10))
            {
                foreach (int j in Enumerable.Range(0, (MNISTHelper.ImageHeight - 1) * 10))
                {
                    var (gradientMagnitudeLowLimitIndex, gradientMagnitudeHighLimitIndex) = GetLimitsIndices(magnitudeAccumulativeDistribution, random, MagnitudeRangesCount);
                    var (gradientAngleLowLimitIndex, gradientAngleHighLimitIndex) = GetLimitsIndices(angleAccumulativeDistribution, random, AngleRangesCount);

                    double gradientAngleLowLimit = Math.PI * gradientAngleLowLimitIndex / 180.0 - Math.PI;
                    double gradientAngleHighLimit = gradientAngleLowLimit + 2 * Math.PI / AngleRangesCount;
                    if (gradientAngleHighLimit > Math.PI)
                        gradientAngleHighLimit -= 2 * Math.PI;

                    Detector detector = new()
                    {
                        CenterX = i / 10.0,
                        CenterY = j / 10.0,
                        GradientMagnitudeLowLimit = gradientMagnitudeLowLimitIndex,
                        GradientMagnitudeHighLimit = gradientMagnitudeHighLimitIndex,
                        GradientAngleLowLimit = gradientAngleLowLimit,
                        GradientAngleHighLimit = gradientAngleHighLimit,
                    };                    
                    detectors.Add(detector);
                }                
            }

            GradientInPoint[,] gradientMatrix2 = SobelOperator.ApplySobel(images[2], MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);
            List<Detector> activatedDetectors = detectors.Where(d => d.IsActivated(gradientMatrix2)).ToList();
            // Применяем оператор Собеля к первому изображению
            var originalBitmap = MNISTHelper.GetBitmap(images[2], MNISTHelper.ImageWidth, MNISTHelper.ImageHeight);            
            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix2);
            var gradientBigBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix2);
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            VisualisationHelper.ShowImages([originalBitmap, gradientBitmap, gradientBigBitmap, detectorsActivationBitmap]);
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
