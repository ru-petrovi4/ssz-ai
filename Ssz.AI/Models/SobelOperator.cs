using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Models
{
    public static class SobelOperator
    {
        public const int MagnitudeMaxValue = 1449;

        // Операторы Собеля для X и Y
        private static int[,] SobelX = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        private static int[,] SobelY = {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };

        public static DenseMatrix<GradientInPoint> ApplySobel(byte[] mnistImageData, int width, int height)
        {
            DenseMatrix<GradientInPoint> gradientMatrix = new DenseMatrix<GradientInPoint>(width, height);

            for (int y = 1; y < height - 1; y += 1)
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    // Применяем фильтры Собеля для X и Y
                    int gradX = 0;
                    int gradY = 0;

                    for (int i = -1; i <= 1; i += 1)
                    {
                        for (int j = -1; j <= 1; j += 1)
                        {
                            gradX += SobelX[i + 1, j + 1] * mnistImageData[x + j + (y + i) * width];
                            gradY += SobelY[i + 1, j + 1] * mnistImageData[x + j + (y + i) * width];
                        }
                    }

                    // Вычисляем магнитуду и угол градиента
                    double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                    // [-pi, pi]
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах

                    GradientInPoint gradientInPoint = new()
                    {
                        GradX = gradX,
                        GradY = gradY,
                        Magnitude = magnitude,
                        Angle = angle,
                    };

                    gradientMatrix[x, y] = gradientInPoint;
                }
            }

            return gradientMatrix;
        }

        public static DenseMatrix<GradientInPoint> ApplySobel_Simplified(byte[] mnistImageData, int width, int height)
        {
            DenseMatrix<GradientInPoint> gradientMatrix = new DenseMatrix<GradientInPoint>(width, height);
            
            int cx = width / 2;
            int cy = height / 2;

            int gradX = 0;
            int gradY = 0;

            for (int i = -1; i <= 1; i += 1)
            {
                for (int j = -1; j <= 1; j += 1)
                {
                    gradX += SobelX[i + 1, j + 1] * mnistImageData[cx + j + (cy + i) * width];
                    gradY += SobelY[i + 1, j + 1] * mnistImageData[cx + j + (cy + i) * width];
                }
            }

            // Вычисляем магнитуду и угол градиента
            double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
            // [-pi, pi]
            double angle = Math.Atan2(gradY, gradX); // Угол в радианах

            GradientInPoint gradientInPoint = new()
            {
                GradX = gradX,
                GradY = gradY,
                Magnitude = magnitude,
                Angle = angle,
            };

            for (int y = 1; y < height - 1; y += 1)
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    gradientMatrix[x, y] = gradientInPoint;
                }
            }

            return gradientMatrix;
        }

        public static DenseMatrix<GradientInPoint> ApplySobel_Simplified2(byte[] mnistImageData, int width, int height, double magnitude, double angle)
        {
            DenseMatrix<GradientInPoint> gradientMatrix = new DenseMatrix<GradientInPoint>(width, height);

            int gradX = 0;
            int gradY = 0;                        

            gradX = (int)(Math.Cos(angle) * magnitude);
            gradY = (int)(Math.Sin(angle) * magnitude);

            GradientInPoint gradientInPoint = new()
            {
                GradX = gradX,
                GradY = gradY,
                Magnitude = magnitude,
                Angle = angle,
            };

            for (int y = 1; y < height - 1; y += 1)
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    gradientMatrix[x, y] = gradientInPoint;
                }
            }

            return gradientMatrix;
        }

        public static DenseMatrix<GradientInPoint> ApplySobel(Bitmap bitmap, int width, int height)
        {
            DenseMatrix<GradientInPoint> gradientMatrix = new DenseMatrix<GradientInPoint>(width, height);

            for (int y = 1; y < height - 1; y += 1)
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    // Применяем фильтры Собеля для X и Y
                    int gradX = 0;
                    int gradY = 0;

                    for (int i = -1; i <= 1; i += 1)
                    {
                        for (int j = -1; j <= 1; j += 1)
                        {
                            Color pixelColor = bitmap.GetPixel(x + j, y + i);

                            // Вычисляем яркость (от 0.0 до 1.0)
                            float brightness = pixelColor.GetBrightness();

                            int brightnessByte = (int)(brightness * 255);

                            gradX += SobelX[i + 1, j + 1] * brightnessByte;
                            gradY += SobelY[i + 1, j + 1] * brightnessByte;
                        }
                    }

                    // Вычисляем магнитуду и угол градиента
                    double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                    // [-pi, pi]
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах

                    GradientInPoint gradientInPoint = new()
                    {
                        GradX = gradX,
                        GradY = gradY,
                        Magnitude = magnitude,
                        Angle = angle,
                    };

                    gradientMatrix[x, y] = gradientInPoint;
                }
            }

            return gradientMatrix;
        }

        public static GradientInPoint[,] ApplySobelObsoslete(byte[] mnistImageData, int width, int height)
        {
            GradientInPoint[,] gradientMatrix = new GradientInPoint[width, height];

            for (int y = 1; y < height - 1; y += 1)                
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    // Применяем фильтры Собеля для X и Y
                    int gradX = 0;
                    int gradY = 0;

                    for (int i = -1; i <= 1; i += 1)
                    {
                        for (int j = -1; j <= 1; j += 1)
                        {
                            gradX += SobelX[i + 1, j + 1] * mnistImageData[x + j + (y + i) * width];
                            gradY += SobelY[i + 1, j + 1] * mnistImageData[x + j + (y + i) * width];
                        }
                    }                    

                    // Вычисляем магнитуду и угол градиента
                    double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                    // [-pi, pi]
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах

                    GradientInPoint gradientInPoint = new()
                    {
                        GradX = gradX,
                        GradY = gradY,
                        Magnitude = magnitude,
                        Angle = angle,
                    };

                    gradientMatrix[x, y] = gradientInPoint;
                }
            }

            return gradientMatrix;
        }

        public static GradientInPoint[,] ApplySobelObsoslete(Bitmap bitmap, int width, int height)
        {
            GradientInPoint[,] gradientMatrix = new GradientInPoint[width, height];

            for (int y = 1; y < height - 1; y += 1)                
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    // Применяем фильтры Собеля для X и Y
                    int gradX = 0;
                    int gradY = 0;

                    for (int i = -1; i <= 1; i += 1)
                    {
                        for (int j = -1; j <= 1; j += 1)
                        {
                            Color pixelColor = bitmap.GetPixel(x + j, y + i);

                            // Вычисляем яркость (от 0.0 до 1.0)
                            float brightness = pixelColor.GetBrightness();

                            int brightnessByte = (int)(brightness * 255);

                            gradX += SobelX[i + 1, j + 1] * brightnessByte;
                            gradY += SobelY[i + 1, j + 1] * brightnessByte;
                        }
                    }

                    // Вычисляем магнитуду и угол градиента
                    double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                    // [-pi, pi]
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах

                    GradientInPoint gradientInPoint = new()
                    {
                        GradX = gradX,
                        GradY = gradY,
                        Magnitude = magnitude,
                        Angle = angle,
                    };

                    gradientMatrix[x, y] = gradientInPoint;
                }
            }

            return gradientMatrix;
        }

        public static void CalculateDistribution(DenseMatrix<GradientInPoint> gradientMatrix, GradientDistribution gradientDistribution, ICortexConstants constants)
        {
            int width = gradientMatrix.Dimensions[0];
            int height = gradientMatrix.Dimensions[1];

            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    var magnitudeInt = (int)gradientMatrix[x, y].Magnitude;
                    if (magnitudeInt < constants.DetectorMinGradientMagnitude)
                        continue;

                    gradientDistribution.MagnitudeData[magnitudeInt] += 1;

                    int angleDegree = (int)(180.0 * gradientMatrix[x, y].Angle / Math.PI + 180.0);
                    if (angleDegree == 360)
                        angleDegree = 0;
                    gradientDistribution.AngleData[angleDegree] += 1;
                }
            }

            //var detectors = new DenseMatrix<Detector>((int)(gradientMatrix.Dimensions[0] / constants.DetectorDelta), (int)(gradientMatrix.Dimensions[1] / constants.DetectorDelta));
            //foreach (int dy in Enumerable.Range(0, detectors.Dimensions[1]))
            //    foreach (int dx in Enumerable.Range(0, detectors.Dimensions[0]))
            //    {
            //        (double magnitude, double angle) = MathHelper.GetInterpolatedGradient(dx * constants.DetectorDelta, dy * constants.DetectorDelta, gradientMatrix);

            //        var magnitudeInt = (int)magnitude;
            //        if (magnitudeInt < Detector.GradientMagnitudeMinimum)
            //            continue;

            //        gradientDistribution.MagnitudeData[magnitudeInt] += 1;

            //        int angleDegree = (int)(180.0 * angle / Math.PI + 180.0);
            //        if (angleDegree == 360)
            //            angleDegree = 0;
            //        gradientDistribution.AngleData[angleDegree] += 1;
            //    }            
        }

        public static void CalculateDistributionObsolete(GradientInPoint[,] gradientMatrix, ICortexConstants constants, GradientDistribution gradientDistribution)
        {
            int width = gradientMatrix.GetLength(0);
            int height = gradientMatrix.GetLength(1);

            for (int y = 0; y < height; y += 1)                
            {
                for (int x = 0; x < width; x += 1)
                {
                    var magnitudeInt = (int)gradientMatrix[x, y].Magnitude;
                    if (magnitudeInt < constants.DetectorMinGradientMagnitude)
                        continue;

                    gradientDistribution.MagnitudeData[magnitudeInt] += 1;

                    int angleDegree = (int)(180.0 * gradientMatrix[x, y].Angle / Math.PI + 180.0);
                    if (angleDegree == 360)
                        angleDegree = 0;
                    gradientDistribution.AngleData[angleDegree] += 1;
                }
            }
        }             
    }
}