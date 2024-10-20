using Ssz.AI.Grafana;
using System;
using System.DrawingCore;
using System.IO;

namespace Ssz.AI.Models
{
    public static class SobelOperator
    {
        public const int MagnitudeMaxValue = 1449;

        // Операторы Собеля для X и Y
        private static int[,] sobelX = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        private static int[,] sobelY = {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };

        public static GradientInPoint[,] ApplySobel(byte[] mnistImageData, int width, int height)
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
                            gradX += sobelX[i + 1, j + 1] * mnistImageData[x + j + (y + i) * width];
                            gradY += sobelY[i + 1, j + 1] * mnistImageData[x + j + (y + i) * width];
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

        public static GradientInPoint[,] ApplySobel(Bitmap bitmap, int width, int height)
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

                            gradX += sobelX[i + 1, j + 1] * brightnessByte;
                            gradY += sobelY[i + 1, j + 1] * brightnessByte;
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

        public static void CalculateDistribution(GradientInPoint[,] gradientMatrix, GradientDistribution gradientDistribution)
        {
            int width = gradientMatrix.GetLength(0);
            int height = gradientMatrix.GetLength(1);

            for (int y = 0; y < height; y += 1)                
            {
                for (int x = 0; x < width; x += 1)
                {
                    var magnitudeInt = (int)gradientMatrix[x, y].Magnitude;
                    if (magnitudeInt < Detector.GradientMagnitudeMinimum)
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