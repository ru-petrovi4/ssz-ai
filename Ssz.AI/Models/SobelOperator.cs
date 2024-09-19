using Ssz.AI.Grafana;
using System;
using System.DrawingCore;
using System.IO;

namespace Ssz.AI.Models
{
    public static class SobelOperator
    {
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

        public static (Bitmap originalBitmap, Bitmap gradientBitmap) ApplySobel(byte[] mnistImageData, int width, int height)
        {            
            Bitmap originalBitmap = new Bitmap(width, height);

            // Проходим по каждому пикселю и устанавливаем его в Bitmap
            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    // Значение пикселя из массива байтов
                    byte pixelValue = mnistImageData[x + y * width];

                    // Преобразуем значение пикселя в оттенок серого (0-255)
                    Color color = Color.FromArgb(pixelValue, pixelValue, pixelValue);

                    // Устанавливаем пиксель в изображении
                    originalBitmap.SetPixel(x, y, color);
                }
            }


            Bitmap gradientBitmap = new Bitmap(width, height);

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
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах

                    // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                    double normalizedAngle = (angle + Math.PI) / (2 * Math.PI);

                    // Преобразуем магнитуду в яркость
                    int brightness = (int)(255 * magnitude / 1448.0); // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))

                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    Color color = ColorFromHSV(360 * normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return (originalBitmap, gradientBitmap);
        }

        public static void ApplySobel(byte[] mnistImageData, int width, int height, GradientDistribution gradientDistribution)
        {
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
                    double angle = Math.Atan2(gradY, gradX); // Угол в радианах

                    gradientDistribution.Data[(int)magnitude] += 1;                    
                }
            }
        }

        // Преобразование HSV в RGB (используется для цветового кодирования угла градиента)
        public static Color ColorFromHSV(double hue, double saturation, double value)
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