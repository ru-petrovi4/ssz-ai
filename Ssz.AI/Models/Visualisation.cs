using Ssz.AI.Helpers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;

namespace Ssz.AI.Models
{
    public static class Visualisation
    {
        public static Bitmap GetGradientBitmap(DenseMatrix<GradientInPoint> gradientMatrix)
        {
            int width = gradientMatrix.Dimensions[0];
            int height = gradientMatrix.Dimensions[1];

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
                    Color color = ColorFromHSV(normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return gradientBitmap;
        }

        public static (DenseMatrix<GradientInPoint>, Bitmap) GetGeneratedLine_GradientMatrix(int width, int height, double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            var Generated_CenterXDelta = (int)(positionK * width / 2.0);
            var Generated_CenterX = (int)(width / 2.0) + Generated_CenterXDelta;
            var Generated_CenterY = (int)(height / 2.0);

            var Generated_AngleDelta = angleK * 2.0 * Math.PI;
            var Generated_Angle = Math.PI / 2 + Generated_AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(Generated_CenterX + lineLength * Math.Cos(Generated_Angle));
            int endY = (int)(Generated_CenterY + lineLength * Math.Sin(Generated_Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(Generated_CenterX - lineLength * Math.Cos(Generated_Angle));
            int startY = (int)(Generated_CenterY - lineLength * Math.Sin(Generated_Angle));

            Bitmap originalBitmap = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(originalBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Создаем кисть и устанавливаем толщину линии
                using (Pen pen = new Pen(Color.White, 15))
                {
                    // Рисуем наклонную линию
                    g.DrawLine(pen, startX, startY, endX, endY);
                }
            }

            // Уменьшаем изображение до размера 28x28

            // Создаем пустое изображение 28x28

            int smallWidth = MNISTHelper.MNISTImageWidthPixels;
            int smallHeight = MNISTHelper.MNISTImageHeightPixels;

            Bitmap resizedBitmap = new Bitmap(smallWidth, smallHeight);
            using (Graphics g = Graphics.FromImage(resizedBitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);

                // Настраиваем высококачественные параметры для уменьшения
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.CompositingQuality = CompositingQuality.HighQuality;

                // Масштабируем изображение
                g.DrawImage(originalBitmap, new Rectangle(0, 0, smallWidth, smallHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению            
            return (SobelOperator.ApplySobel(resizedBitmap, smallWidth, smallHeight), resizedBitmap);
        }

        public static Bitmap GetBitmap(List<Detector> activatedDetectors)
        {
            int width = MNISTHelper.MNISTImageWidthPixels * 10;
            int height = MNISTHelper.MNISTImageHeightPixels * 10;

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
                    Color color = ColorFromHSV(normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return gradientBitmap;
        }

        public static Bitmap? GetContextSyncingMatrixFloatBitmap(
            MatrixFloat? contextSyncingMatrixFloat,
            int? syncingMatrixFloat_TrainingCount)
        {
            if (contextSyncingMatrixFloat is null)
                return null;            

            Bitmap bitmap = new Bitmap(contextSyncingMatrixFloat.Dimensions[1], contextSyncingMatrixFloat.Dimensions[0]);
            
            try
            {
                foreach (int i in Enumerable.Range(0, contextSyncingMatrixFloat.Dimensions[0]))
                {
                    //float max = 0.1f;
                    float max = Single.MinValue;
                    foreach (int j in Enumerable.Range(0, contextSyncingMatrixFloat.Dimensions[1]))
                    {
                        float v = contextSyncingMatrixFloat[i, j];
                        if (v > max)
                            max = v;
                    }

                    foreach (int j in Enumerable.Range(0, contextSyncingMatrixFloat.Dimensions[1]))
                    {
                        int brightness;
                        if (max > 0)
                        {
                            float v = contextSyncingMatrixFloat[i, j];
                            brightness = (int)(255 * v / max);
                            if (brightness < 0)
                                brightness = 0;                            
                        }
                        else
                        {
                            brightness = 0;
                        }
                        bitmap.SetPixel(j, i, Color.FromArgb(brightness, brightness, brightness));
                    }
                }
            }
            catch
            {
            }            

            return bitmap;
        }

        public static Bitmap? GetContextSyncingMatrixFloatBitmap2(
            MatrixFloat? contextSyncingMatrixFloat,
            int? syncingMatrixFloat_TrainingCount)
        {
            if (contextSyncingMatrixFloat is null)
                return null;

            Bitmap bitmap = new Bitmap(contextSyncingMatrixFloat.Dimensions[1], contextSyncingMatrixFloat.Dimensions[0]);
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);
            }

            try
            {
                foreach (int i in Enumerable.Range(0, contextSyncingMatrixFloat.Dimensions[0]))
                {
                    int jMax = 0;
                    float max = contextSyncingMatrixFloat[i, 0];    
                    
                    //foreach (int j in Enumerable.Range(0, contextSyncingMatrixFloat.Dimensions[1]))
                    //{
                    //    float v = contextSyncingMatrixFloat[i, j];
                    //    if (v > max)
                    //        max = v;
                    //}

                    foreach (int j in Enumerable.Range(0, contextSyncingMatrixFloat.Dimensions[1]))
                    {
                        float v = contextSyncingMatrixFloat[i, j];                        
                        if (v > max)
                        {
                            max = v;
                            jMax = j;                            
                        }                        
                    }

                    bitmap.SetPixel(jMax, i, Color.FromArgb(255, 255, 255));
                }
            }
            catch
            {

            }

            return bitmap;
        }

        public static Bitmap GetMiniColumsActivityBitmap_Obsolete(Cortex cortex, ActivitiyMaxInfo activitiyMaxInfo)
        {
            var miniColumns = cortex.MiniColumns;            

            int width = miniColumns.Dimensions[0];
            int height = miniColumns.Dimensions[1];            

            double maxActivity = Double.MinValue;
            double minActivity = Double.MaxValue;
            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    Cortex.MiniColumn mc = miniColumns[x, y];
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

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    Cortex.MiniColumn mc = miniColumns[x, y];
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

            foreach (Cortex.MiniColumn? maxActivityMiniColumn in activitiyMaxInfo.ActivityMax_MiniColumns)
            {
                gradientBitmap.SetPixel(maxActivityMiniColumn.MCX, maxActivityMiniColumn.MCY, Color.Blue);
            }

            Cortex.MiniColumn? maxSuperActivityMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(new Random());
            if (maxSuperActivityMiniColumn is not null)
                gradientBitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.Red);

            return gradientBitmap;
        }        

        public static Bitmap GetGradientBigBitmap(DenseMatrix<GradientInPoint> gradientMatrix)
        {
            int width = gradientMatrix.Dimensions[0] * 10;
            int height = gradientMatrix.Dimensions[1] * 10;

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 0; y < height - 10; y += 1)
            {
                for (int x = 0; x < width - 10; x += 1)
                {
                    GradientInPoint gradientInPoint = MathHelper.GetInterpolatedGradient(x / 10.0, y / 10.0, gradientMatrix);

                    // Преобразуем магнитуду в яркость
                    int brightness = (int)(255 * gradientInPoint.Magnitude / 1448.0); // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))
                    if (brightness > 255)
                        brightness = 255;

                    // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                    double normalizedAngle = (gradientInPoint.Angle + Math.PI) / (2 * Math.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    Color color = ColorFromHSV(normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return gradientBitmap;
        }

        public static Bitmap GetGradientBigBitmapObsolete(GradientInPoint[,] gradientMatrix)
        {
            int width = MNISTHelper.MNISTImageWidthPixels * 10;
            int height = MNISTHelper.MNISTImageHeightPixels * 10;

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 0; y < height - 10; y += 1)
            {
                for (int x = 0; x < width - 10; x += 1)
                {
                    (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(x / 10.0, y / 10.0, gradientMatrix);                    

                    // Преобразуем магнитуду в яркость
                    int brightness = (int)(255 * magnitude / 1448.0); // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))
                    if (brightness > 255)
                        brightness = 255;

                    // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                    double normalizedAngle = (angle + Math.PI) / (2 * Math.PI);
                    // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                    Color color = ColorFromHSV(normalizedAngle, 1, brightness / 255.0);

                    gradientBitmap.SetPixel(x, y, color);
                }
            }

            return gradientBitmap;
        }        

        // Преобразование HSV в RGB (используется для цветового кодирования угла градиента)
        public static Color ColorFromHSV(double hue, double saturation, double value)
        {
            hue *= 360;
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

        public static Bitmap GetBitmapFromMiniColums_ActivityColor(Cortex cortex)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            float activityMin = float.MaxValue;
            float activityMax = float.MinValue;

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && !float.IsNaN(mc.Temp_Activity.Item3))
                    {
                        if (mc.Temp_Activity.Item3 > activityMax)
                            activityMax = mc.Temp_Activity.Item3;
                        if (mc.Temp_Activity.Item3 < activityMin)
                            activityMin = mc.Temp_Activity.Item3;
                    }
                }

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && !float.IsNaN(mc.Temp_Activity.Item3))
                    {
                        if (activityMax > activityMin)
                        {
                            if (mc.Temp_Activity.Item3 == activityMax)                            
                            {
                                bitmap.SetPixel(mcx, mcy, Color.White);                                
                            }
                            else
                            {
                                int brightness = (int)(255 * (mc.Temp_Activity.Item3 - activityMin) / (activityMax - activityMin));
                                bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, brightness, 0));
                            }
                        }
                        else
                        {
                            bitmap.SetPixel(mcx, mcy, Color.White);
                        }
                    }
                    else
                    {
                        bitmap.SetPixel(mcx, mcy, Color.Black);
                    }
                }            

            return bitmap;
        }

        public static Bitmap GetBitmapFromMiniColums_SuperActivityColor(Cortex cortex, ActivitiyMaxInfo? activitiyMaxInfo)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            float superActivityMin = float.MaxValue;
            float superActivityMax = float.MinValue;

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && !float.IsNaN(mc.Temp_SuperActivity))
                    {
                        if (mc.Temp_SuperActivity > superActivityMax)
                            superActivityMax = mc.Temp_SuperActivity;
                        if (mc.Temp_SuperActivity < superActivityMin)
                            superActivityMin = mc.Temp_SuperActivity;
                    }                    
                }

            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && !float.IsNaN(mc.Temp_SuperActivity))
                    {
                        if (superActivityMax > superActivityMin)
                        {
                            if ((activitiyMaxInfo is not null &&
                                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Contains(mc)) ||
                                    (activitiyMaxInfo is null &&
                                    mc.Temp_SuperActivity == superActivityMax))
                            {
                                if (mc.Temp_Activity.Item2 == 0.0f)
                                    bitmap.SetPixel(mcx, mcy, Color.White);
                                else
                                    bitmap.SetPixel(mcx, mcy, Color.Blue);                                
                            }
                            else
                            {
                                int brightness = (int)(255 * (mc.Temp_SuperActivity - superActivityMin) / (superActivityMax - superActivityMin));
                                bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, 0, 0));
                            }
                        }
                        else
                        {
                            bitmap.SetPixel(mcx, mcy, Color.White);
                        }
                    }
                    else
                    {
                        bitmap.SetPixel(mcx, mcy, Color.Black);
                    }
                }

            //if (allMaxSuperActivity)
            //{
            //    foreach (Cortex.MiniColumn maxSuperActivityMiniColumn in activitiyMaxInfo.SuperActivityMax_MiniColumns)
            //    {
            //        bitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 255, 255));
            //    }
            //}
            //else
            //{
            //    Cortex.MiniColumn? maxSuperActivityMiniColumn = cortex.Temp_SuperActivityMax_MiniColumn;
            //    if (maxSuperActivityMiniColumn is not null)
            //        bitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 255, 255));
            //}

            return bitmap;
        }

        public static Bitmap GetMiniColumsActivityMaxBitmap(Cortex cortex, ActivitiyMaxInfo activitiyMaxInfo, bool allSuperActivity)
        {
            var miniColumns = cortex.MiniColumns;

            int width = miniColumns.Dimensions[0];
            int height = miniColumns.Dimensions[1];

            Bitmap gradientBitmap = new Bitmap(width, height);

            for (int y = 0; y < height; y += 1)
            {
                for (int x = 0; x < width; x += 1)
                {
                    gradientBitmap.SetPixel(x, y, Color.Black);
                }
            }

            foreach (Cortex.MiniColumn maxActivityMiniColumn in activitiyMaxInfo.ActivityMax_MiniColumns)
            {
                gradientBitmap.SetPixel(maxActivityMiniColumn.MCX, maxActivityMiniColumn.MCY, Color.FromArgb(255, 255, 0));
            }

            if (allSuperActivity)
            {
                foreach (Cortex.MiniColumn maxSuperActivityMiniColumn in activitiyMaxInfo.SuperActivityMax_MiniColumns)
                {
                    gradientBitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 0, 0));
                }
            }
            else
            {
                Cortex.MiniColumn? maxSuperActivityMiniColumn = cortex.Temp_SuperActivityMax_MiniColumn;
                if (maxSuperActivityMiniColumn is not null)
                    gradientBitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 0, 0));
            }            

            return gradientBitmap;
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

        public static Bitmap GetBitmapFromMiniColumsMemoriesColor(Cortex cortex)
        {
            Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

            //double normalizedMagnitudeMax = Double.MinValue;
            //foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            //    foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            //    {
            //        var mc = cortex.MiniColumns[mcx, mcy];
            //        if (mc is not null && mc.Memories.Count > 0)
            //        {
            //            double gradX = 0.0;
            //            double gradY = 0.0;
            //            foreach (var memory in mc.Memories)
            //            {
            //                gradX += memory.AverageGradientInPoint.GradX;
            //                gradY += memory.AverageGradientInPoint.GradY;
            //            }
            //            if (mc.Memories.Count > 0)
            //            {
            //                gradX /= mc.Memories.Count;
            //                gradY /= mc.Memories.Count;
            //            }
            //            double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
            //            double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

            //            // Преобразуем магнитуду в яркость
            //            double normalizedMagnitude = magnitude / 1448.0; // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))                        

            //            if (normalizedMagnitude > normalizedMagnitudeMax)
            //                normalizedMagnitudeMax = normalizedMagnitude;
            //        }                    
            //    }

            using (Graphics g = Graphics.FromImage(bitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);
            }
            
            foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
                foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
                {
                    var mc = cortex.MiniColumns[mcx, mcy];
                    if (mc is not null && mc.Memories.Count > 0)
                    {                        
                        double gradX = 0.0;
                        double gradY = 0.0;

                        foreach (var memory in mc.Memories)
                        {
                            if (memory is null)
                                continue;
                            gradX += memory.PictureAverageGradientInPoint.GradX;
                            gradY += memory.PictureAverageGradientInPoint.GradY;
                        }
                        if (mc.Memories.Count > 0)
                        {
                            gradX /= mc.Memories.Count;
                            gradY /= mc.Memories.Count;
                        }
                        double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);
                        double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

                        // Преобразуем магнитуду в яркость
                        double normalizedMagnitude = magnitude / cortex.Constants.GeneratedMaxGradientMagnitude; // 1448 - максимальная теоретическая магнитуда Собеля для 8-битных изображений (255 * sqrt(2))                        
                        //brightness = 0.5 + (1 - brightness) * 0.5;
                        double saturation = 0.3 + normalizedMagnitude;
                        if (saturation > 1)
                            saturation = 1;

                        // Преобразуем угол из диапазона [-pi, pi] в диапазон [0, 1] для цвета
                        double normalizedAngle = (angle + Math.PI) / (2 * Math.PI);
                        // Получаем цвет на основе угла градиента (можно использовать HSV, здесь упрощенный пример через цветовой спектр)
                        Color color = ColorFromHSV(normalizedAngle, saturation, 1);

                        bitmap.SetPixel(mcx, mcy, color);
                    }                    
                }

            //using (Graphics g = Graphics.FromImage(bitmap))
            //{
            //    // Устанавливаем черный фон
            //    g.DrawEllipse(new Pen(Color.White), maxMemoryMiniColumn.MCX - 3, maxMemoryMiniColumn.MCY - 3, 5, 5);
            //}

            return bitmap;
        }

        public static Bitmap GetBigMatrixFloatBitmap(MatrixFloat bigMatrixFloat, int j)
        {            
            Bitmap bitmap = new Bitmap((int)(bigMatrixFloat.Dimensions[1]), (int)(bigMatrixFloat.Dimensions[0]));

            using (Graphics g = Graphics.FromImage(bitmap))
            {
                // Устанавливаем черный фон
                g.Clear(Color.Black);
            }

            foreach (int mcx in Enumerable.Range(0, bigMatrixFloat.Dimensions[1]))
            {
                float min = float.MaxValue;
                float max = float.MinValue;

                var columnFloat = bigMatrixFloat.GetColumn(mcx);

                foreach (int mcy in Enumerable.Range(0, columnFloat.Length))
                {
                    var v = columnFloat[mcy];
                    if (v > max)
                        max = v;
                    if (v < min)
                        min = v;
                }                

                if (max != min)
                {
                    foreach (int mcy in Enumerable.Range(0, columnFloat.Length))
                    {
                        var v = columnFloat[mcy];

                        int brightness = (int)(255 * ((v - min) / (max - min)));

                        bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, brightness, brightness));
                    }                    
                }
            } 
            
            //Bitmap resizedBitmap = new Bitmap(MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
            //using (Graphics g = Graphics.FromImage(resizedBitmap))
            //{
            //    // Устанавливаем черный фон
            //    g.Clear(Color.Black);

            //    // Настраиваем высококачественные параметры для уменьшения
            //    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            //    g.SmoothingMode = SmoothingMode.HighQuality;
            //    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            //    g.CompositingQuality = CompositingQuality.HighQuality;

            //    // Масштабируем изображение
            //    g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            //}

            return bitmap;
        }
    }
}



//public static Bitmap GetBitmapFromMiniColumsMemoriesCount(Cortex cortex)
//{
//    Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

//    int minMemoriesCount = int.MaxValue;
//    int maxMemoriesCount = int.MinValue;

//    foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
//        foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
//        {
//            var mc = cortex.MiniColumns[mcx, mcy];
//            if (mc is not null)
//            {
//                if (mc.Memories.Count > maxMemoriesCount)
//                    maxMemoriesCount = mc.Memories.Count;
//                if (mc.Memories.Count < minMemoriesCount)
//                    minMemoriesCount = mc.Memories.Count;
//            }
//        }

//    minMemoriesCount = 0;

//    foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
//        foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
//        {
//            var mc = cortex.MiniColumns[mcx, mcy];
//            if (mc is not null && maxMemoriesCount != minMemoriesCount)
//            {
//                int brightness = (int)(255 * ((float)(mc.Memories.Count - minMemoriesCount)) / (maxMemoriesCount - minMemoriesCount));

//                bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, brightness, brightness));
//            }
//            else
//            {
//                bitmap.SetPixel(mcx, mcy, Color.Black);
//            }
//        }

//    return bitmap;
//}
