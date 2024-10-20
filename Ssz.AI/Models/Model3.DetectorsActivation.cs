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
        /// <summary>
        ///     Активация детекторов на тестовом изображении из MNIST
        /// </summary>
        public Model3()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            //List<GradientInPoint[,]> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                //gradientMatricesCollection.Add(gradientMatrix);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            List<Detector> detectors = DetectorsGenerator.Generate(gradientDistribution, AngleRangesCount, MagnitudeRangesCount, HashLength);

            // Применяем оператор Собеля к первому изображению
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(images[2], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
            List<Detector> activatedDetectors = detectors.Where(d => d.GetIsActivated(gradientMatrix)).ToList();            
            var originalBitmap = MNISTHelper.GetBitmap(images[2], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);            
            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);            
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            VisualisationHelper.ShowImages([originalBitmap, gradientBitmap, detectorsActivationBitmap]);
        }

        public const int AngleRangesCount = 4;

        public const int MagnitudeRangesCount = 4;

        public const int HashLength = 200;
    }
}
