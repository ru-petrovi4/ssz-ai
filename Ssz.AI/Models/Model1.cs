using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class Model1
    {
        public Model1()
        {
            int numDetectors = 2000;
            double receptiveWidth = 0.2;

            var detectors = GenerateRandomCoordinatesAndRanges(numDetectors, new Size(28, 28), receptiveWidth);
            var visualization = new DetectorVisualization(detectors);
            var gifImages = GenerateGif();
            visualization.Visualize(gifImages);
        }

        public static List<Detector> GenerateRandomCoordinatesAndRanges(int numDetectors, Size imageShape, double receptiveWidth)
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
                    detectors.Add(new Detector(centerX, centerY, receptiveWidth, angleRange, gradientMagnitudeRange));
                }
            }

            return detectors;
        }

        public static List<Mat> GenerateGif()
        {
            var images = new List<Mat>();
            var image = new Mat(new Size(280, 280), MatType.CV_8UC1, Scalar.Black);
            Cv2.Line(image, new Point(15, 0), new Point(15, 200), Scalar.White, 2);
            var smallImage = new Mat();
            Cv2.Resize(image, smallImage, new Size(28, 28));

            for (int i = 0; i < 100; i++)
            {
                var matExpr = Mat.Eye(rows: 2, cols: 3, MatType.CV_32F);
                // TODO
                //matExpr.Set(0, 2, i * 0.1f);
                var shiftedImage = new Mat();
                Cv2.WarpAffine(smallImage, shiftedImage, matExpr, new Size(28, 28));
                images.Add(shiftedImage);
            }

            return images;
        }
    }
}
