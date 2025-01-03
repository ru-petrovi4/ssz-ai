using Ssz.AI.Helpers;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.Linq;

namespace Ssz.AI.Models
{
    public static class Visualisation
    {
        public static Bitmap GetBitmap(List<Detector> activatedDetectors)
        {
            int width = MNISTHelper.MNISTImageWidth * 10;
            int height = MNISTHelper.MNISTImageHeight * 10;

            Bitmap bitmap = new Bitmap(width, height);

            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    bitmap.SetPixel(x, y, Color.Black);
                }
            }

            foreach (var detector in activatedDetectors)
            {
                bitmap.SetPixel((int)(detector.CenterX * 10.0), (int)(detector.CenterY * 10.0), Color.FromArgb(255, 200, 200, 200));
            }

            return bitmap;
        }

        public static Bitmap GetGradientBitmap(GradientInPoint[,] gradientMatrix)
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

        public static Bitmap GetMiniColumsActivityBitmap(Cortex_WithSubarea cortex, MiniColumnsActivity.ActivitiyMaxInfo activitiyMaxInfo)
        {
            var random = new Random();

            var miniColumns = cortex.MiniColumns;            

            int width = miniColumns.Dimensions[0];
            int height = miniColumns.Dimensions[1];

            Bitmap gradientBitmap = new Bitmap(width, height);

            double maxActivity = Double.MinValue;
            double minActivity = Double.MaxValue;
            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    Cortex_WithSubarea.MiniColumn mc = miniColumns[x, y];
                    if (mc is not null)
                    {
                        float a = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
                        if (a > maxActivity)
                            maxActivity = a;
                        if (a < minActivity)
                            minActivity = a;
                    }
                }
            }

            //minActivity = minActivity + (maxActivity - minActivity) * 0.66;            
            minActivity = 0.0;

            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    Cortex_WithSubarea.MiniColumn mc = miniColumns[x, y];
                    if (mc is null || maxActivity == minActivity || float.IsNaN(mc.Temp_Activity.Item1) || (mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2) <= minActivity) // || mc.Temp_Activity < miniColumnMinimumActivity
                    {
                        gradientBitmap.SetPixel(x, y, Color.Black);
                    }
                    else
                    {   
                        int brightness = (int)(255 * (mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2 - minActivity) / (maxActivity - minActivity));

                        gradientBitmap.SetPixel(x, y, Color.FromArgb(brightness, brightness, brightness));
                    }
                }
            }

            Cortex_WithSubarea.MiniColumn? maxActivityMiniColumn = activitiyMaxInfo.GetActivityMax_MiniColumn(random);
            if (maxActivityMiniColumn is not null)
                gradientBitmap.SetPixel(maxActivityMiniColumn.MCX, maxActivityMiniColumn.MCY, Color.Red);

            Cortex_WithSubarea.MiniColumn? maxSuperActivityMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
            if (maxSuperActivityMiniColumn is not null)
                gradientBitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.Blue);

            return gradientBitmap;
        }

        public static Bitmap GetGradientBigBitmap(GradientInPoint[,] gradientMatrix)
        {
            int width = MNISTHelper.MNISTImageWidth * 10;
            int height = MNISTHelper.MNISTImageHeight * 10;

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 0; y < height - 10; y += 1)
            {
                for (int x = 0; x < width - 10; x += 1)
                {
                    (double magnitude, double angle) = MathHelper.GetInterpolatedGradient(x / 10.0, y / 10.0, gradientMatrix);                    

                    // Преобразуем магнитуду в яркость
                    int brightness = (int)(255 * magnitude / 1448.0); // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))
                    if (brightness > 255)
                        brightness = 255;

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

        public static Bitmap GetBitmapFromMiniColums_ActivityColor(Cortex_WithSubarea cortex)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null)
                    {
                        bitmap.SetPixel(mcx, mcy, mc.Temp_ActivityColor);
                    }
                    else
                    {
                        bitmap.SetPixel(mcx, mcy, Color.Black);
                    }
                }

            return bitmap;
        }

        public static Bitmap GetBitmapFromMiniColums_SuperActivityColor(Cortex_WithSubarea cortex)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null)
                    {
                        bitmap.SetPixel(mcx, mcy, mc.Temp_SuperActivityColor);
                    }
                    else
                    {
                        bitmap.SetPixel(mcx, mcy, Color.Black);
                    }
                }

            return bitmap;
        }

        public static Bitmap GetBitmapFromMiniColumsMemoriesCount(Cortex_WithSubarea cortex)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            int minMemoriesCount = int.MaxValue;
            int maxMemoriesCount = int.MinValue;

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null)
                    {
                        if (mc.Memories.Count > maxMemoriesCount)
                            maxMemoriesCount = mc.Memories.Count;
                        if (mc.Memories.Count < minMemoriesCount)
                            minMemoriesCount = mc.Memories.Count;
                    }                    
                }

            minMemoriesCount = 0;

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && maxMemoriesCount != minMemoriesCount)
                    {
                        int brightness = (int)(255 * ((float)(mc.Memories.Count - minMemoriesCount)) / (maxMemoriesCount - minMemoriesCount));

                        bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, brightness, brightness));
                    }
                    else
                    {
                        bitmap.SetPixel(mcx, mcy, Color.Black);
                    }
                }

            return bitmap;
        }

        public static Bitmap GetBitmapFromMiniColumsMemoriesCount(Cortex cortex)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            int minMemoriesCount = int.MaxValue;
            int maxMemoriesCount = int.MinValue;

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null)
                    {
                        if (mc.Memories.Count > maxMemoriesCount)
                            maxMemoriesCount = mc.Memories.Count;
                        if (mc.Memories.Count < minMemoriesCount)
                            minMemoriesCount = mc.Memories.Count;
                    }
                }

            minMemoriesCount = 0;

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && maxMemoriesCount != minMemoriesCount)
                    {
                        int brightness = (int)(255 * ((float)(mc.Memories.Count - minMemoriesCount)) / (maxMemoriesCount - minMemoriesCount));

                        bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, brightness, brightness));
                    }
                    else
                    {
                        bitmap.SetPixel(mcx, mcy, Color.Black);
                    }
                }

            return bitmap;
        }
    }
}
