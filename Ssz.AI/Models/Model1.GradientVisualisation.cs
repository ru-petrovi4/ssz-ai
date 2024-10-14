using Avalonia.Controls;
using Avalonia.Layout;
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
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            // Применяем оператор Собеля к первому изображению
            var originalBitmap = MNISTHelper.GetBitmap(images[2], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(images[2], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);
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

            VisualisationHelper.ShowImages([originalBitmap, gradientBitmap]);
            //visualization.Visualize(gifImages);
        }

        public static List<Detector> GenerateDetectors(int numDetectors, Size imageShape, double receptiveWidth)
        {
            var detectors = new List<Detector>();
            var rand = new Random();

            for (int i = 0; i < numDetectors; i++)
            {
                double centerX = rand.NextDouble() * imageShape.Width;
                double centerY = rand.NextDouble() * imageShape.Height;
                double angleRange = rand.NextDouble() * Math.PI;
                double gradientMagnitudeRange = rand.NextDouble() * 255;

                bool isOverlap = detectors.Exists(d =>
                    Math.Sqrt(Math.Pow(centerX - d.CenterX, 2) + Math.Pow(centerY - d.CenterY, 2)) < receptiveWidth);

                if (!isOverlap)
                {
                    //detectors.Add(new Detector(centerX, centerY, receptiveWidth, angleRange, gradientMagnitudeRange));
                }
            }

            return detectors;
        }

        //public static List<Mat> GenerateGif()
        //{
        //    var images = new List<Mat>();
        //    var image = new Mat(new Size(280, 280), MatType.CV_8UC1, Scalar.Black);
        //    Cv2.Line(image, new Point(15, 0), new Point(15, 200), Scalar.White, 2);
        //    var smallImage = new Mat();
        //    Cv2.Resize(image, smallImage, new Size(28, 28));

        //    // TEMPCODE
        //    images.Add(smallImage);
        //    //for (int i = 0; i < 100; i++)
        //    //{
        //    //    var matExpr = Mat.Eye(rows: 2, cols: 3, MatType.CV_32F);
        //    //    // TODO
        //    //    //matExpr.Set(0, 2, i * 0.1f);
        //    //    var shiftedImage = new Mat();
        //    //    Cv2.WarpAffine(smallImage, shiftedImage, matExpr, new Size(28, 28));
        //    //    images.Add(shiftedImage);
        //    //}

        //    return images;
        //}        
    }
}
