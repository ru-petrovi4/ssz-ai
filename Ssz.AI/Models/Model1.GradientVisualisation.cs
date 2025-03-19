using Avalonia.Controls;
using Avalonia.Layout;
using NumSharp;
using OpenCvSharp;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    /// <summary>
    ///     Визуализация градиентов для изображений
    /// </summary>
    public class Model1
    {
        public Model1()
        {   
            NDArray images = np.load("Images(500x500).npy");
            NDArray writerInfo = np.load("WriterInfo.npy");            

            // Применяем оператор Собеля к первому изображению
            //var originalBitmap = HWDDHelper.GetBitmap(images, 1);
            //DenseMatrix<GradientInPoint> gradientMatrix = SobelOperator.ApplySobel(originalBitmap, HWDDHelper.HWDDImageWidthPixels, HWDDHelper.HWDDImageHeightPixels);
            //var gradientBitmap = Visualisation.GetGradientBitmap(gradientMatrix);


            // Сохраняем результат
            //gradientImage.Save("sobel_gradient_image.png");

            //// Выводим информацию о первом изображении
            //Console.WriteLine($"Метка первого изображения: {labels[0]}");
            //for (int i = 0; i < 28; i++)
            //{
            //    for (int j = 0; j < 28; j++)
            //    {
            //        Console.Write(images[0, i * 28 + j] > 0 ? "#" : ".");
            //    }
            //    Console.WriteLine();
            //}            

            //var detectors = GenerateDetectors(numDetectors: 2000, imageShape: new Size(28, 28), receptiveWidth: 0.2);

            //var visualization = new DetectorVisualization(detectors);
            //var gifImages = GenerateGif();

            //VisualisationHelper.ShowImages([originalBitmap, gradientBitmap]);
            //visualization.Visualize(gifImages);
        }     
    }
}
