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
                GradientInPoint[,] gm = SobelOperator.ApplySobelObsoslete(images[i], MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
                //gradientMatricesCollection.Add(gradientMatrix);
                SobelOperator.CalculateDistributionObsolete(gm, gradientDistribution);
            }

            //Retina retina = new(gradientDistribution, AngleRangesCount, MagnitudeRangesCount, HashLength);

            //// Применяем оператор Собеля к первому изображению
            //GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(images[2], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            //List<Detector> activatedDetectors = new List<Detector>(retina.Detectors.Dimensions[0] * retina.Detectors.Dimensions[1]);
            //foreach (int dy in Enumerable.Range(0, retina.Detectors.Dimensions[1]))
            //    foreach (int dx in Enumerable.Range(0, retina.Detectors.Dimensions[0]))
            //    {
            //        Detector d = retina.Detectors[dx, dy];
            //        if (d.GetIsActivated(gradientMatrix))
            //            activatedDetectors.Add(d);
            //    }

            //var originalBitmap = MNISTHelper.GetBitmap(images[2], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);            
            //var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);            
            //var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            ////DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            //VisualisationHelper.ShowImages([originalBitmap, gradientBitmap, detectorsActivationBitmap]);
        }

        public const int AngleRangesCount = 4;

        public const int MagnitudeRangesCount = 4;

        public const int HashLength = 200;
    }
}
