﻿using Ssz.AI.Helpers;
using System;
using System.Collections.Generic;
using System.DrawingCore;

namespace Ssz.AI.Models
{
    public static class Visualisation
    {
        public static Bitmap GetBitmap(List<Detector> activatedDetectors)
        {
            int width = MNISTHelper.ImageWidth * 10;
            int height = MNISTHelper.ImageHeight * 10;

            Bitmap bitmap = new Bitmap(width, height);

            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    bitmap.SetPixel(x, y, Color.FromArgb(255, 0, 0, 0));
                }
            }

            foreach (var detector in activatedDetectors)
            {
                bitmap.SetPixel((int)(detector.CenterX * 10.0), (int)(detector.CenterY * 10.0), Color.FromArgb(255, 200, 200, 200));
            }

            return bitmap;
        }

        public static Bitmap GetBitmap(GradientInPoint[,] gradientMatrix)
        {
            int width = gradientMatrix.GetLength(0);
            int height = gradientMatrix.GetLength(1);

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 1; y < height - 1; y += 1)
            {
                for (int x = 1; x < width - 1; x += 1)
                {
                    GradientInPoint gradientInPoint = gradientMatrix[x, y];

                    // Преобразуем магнитуду в яркость
                    int brightness = (int)(255 * gradientInPoint.Magnitude / 1448.0); // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))

                    // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                    double normalizedAngle = (gradientInPoint.Angle + Math.PI) / (2 * Math.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    Color color = ColorFromHSV(360 * normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return gradientBitmap;
        }

        internal static Bitmap GetGradientBigBitmap(GradientInPoint[,] gradientMatrix)
        {
            int width = MNISTHelper.ImageWidth * 10;
            int height = MNISTHelper.ImageHeight * 10;

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 0; y < height - 10; y += 1)
            {
                for (int x = 0; x < width - 10; x += 1)
                {
                    (double magnitude, double angle) = MathHelper.GetInterpolatedGradient(x / 10.0, y / 10.0, gradientMatrix);                    

                    // Преобразуем магнитуду в яркость
                    int brightness = (int)(255 * magnitude / 1448.0); // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))

                    // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                    double normalizedAngle = (angle + Math.PI) / (2 * Math.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    Color color = ColorFromHSV(360 * normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return gradientBitmap;
        }        

        // Преобразование HSV в RGB (используется для цветового кодирования угла градиента)
        private static Color ColorFromHSV(double hue, double saturation, double value)
        {
            int hi = Convert.ToInt32(Math.Floor(hue / 60)) % 6;
            double f = hue / 60 - Math.Floor(hue / 60);

            value = value * 255;
            int v = Convert.ToInt32(value);
            int p = Convert.ToInt32(value * (1 - saturation));
            int q = Convert.ToInt32(value * (1 - f * saturation));
            int t = Convert.ToInt32(value * (1 - (1 - f) * saturation));

            if (hi == 0)
                return Color.FromArgb(255, v, t, p);
            else if (hi == 1)
                return Color.FromArgb(255, q, v, p);
            else if (hi == 2)
                return Color.FromArgb(255, p, v, t);
            else if (hi == 3)
                return Color.FromArgb(255, p, q, v);
            else if (hi == 4)
                return Color.FromArgb(255, t, p, v);
            else
                return Color.FromArgb(255, v, p, q);
        }
    }
}