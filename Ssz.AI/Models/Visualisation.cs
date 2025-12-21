using Microsoft.AspNetCore.JsonPatch.Internal;
using Ssz.AI.Helpers;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace Ssz.AI.Models;

public static class Visualisation
{
    public static ImageWithDesc[] VisualizeKSearch()
    {
        Dictionary<float, System.Drawing.Bitmap> info = new();

        try
        {
            int currentWordIndex = 8;
            foreach (var it in CsvHelper.ParseCsvMultiline(@",", File.ReadAllText(Path.Combine("Data", @"CortexVisualisationModel_Model01_Logs.2025.12.19 Full Log.txt"))))
            {
                if (it.Count == 16)
                {
                    float key = MathF.Round(new Any(it[currentWordIndex + 2]).ValueAsSingle(false), 4);
                    if (!info.TryGetValue(key, out var bitmap))
                    {
                        bitmap = new(50, 50);

                        using (Graphics g = Graphics.FromImage(bitmap))
                        {
                            // Устанавливаем черный фон
                            g.Clear(Color.Blue);
                        }

                        info.Add(key, bitmap);
                    }
                    float a = new Any(it[currentWordIndex + 1]).ValueAsSingle(false);
                    int brightness = (int)(255 * (a - 5.9f) / 0.1f);
                    if (brightness < 0)
                        brightness = 0;
                    if (brightness > 255)
                        brightness = 255;
                    bitmap.SetPixel(
                        x: (int)(new Any(it[currentWordIndex + 3]).ValueAsSingle(false) * 200),
                        y: (int)(new Any(it[currentWordIndex + 6]).ValueAsSingle(false) * 200),
                        color: Color.FromArgb(brightness, brightness, brightness)
                        );
                }
            }
        }
        catch
        {
        }

        return info
            .OrderBy(kvp => kvp.Key)
            .Select(kvp => new ImageWithDesc
            {
                Image = BitmapHelper.ConvertImageToAvaloniaBitmap(kvp.Value),
                Desc = $"K[1]: {kvp.Key}"
            })
            .ToArray();
    }

    public static Color GetColorFromDiscreteVector(float[] discreteVector)
    {
        List<Color> colors = new();
        for (int i = 0; i < discreteVector.Length; i += 1)
        {
            if (discreteVector[i] > 0.5f)
                colors.Add(ColorFromHSV((double)i / discreteVector.Length, 1.0, 1.0));
        }
        return GetAverageLABColor(colors);
    }    

    public static Bitmap GetBitmapFromMiniColums_ActivityColor(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
    {
        Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

        float activityMin = float.MaxValue;
        float activityMax = float.MinValue;

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null && !float.IsNaN(mc.Temp_Activity.PositiveActivity) && !float.IsNaN(mc.Temp_Activity.NegativeActivity))
                {
                    float activity = mc.Temp_Activity.PositiveActivity + mc.Temp_Activity.NegativeActivity;
                    if (activity > activityMax)
                        activityMax = activity;
                    if (activity < activityMin)
                        activityMin = activity;
                }
            }

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null && !float.IsNaN(mc.Temp_Activity.PositiveActivity) && !float.IsNaN(mc.Temp_Activity.NegativeActivity))
                {
                    if (activityMax > activityMin)
                    {
                        float activity = mc.Temp_Activity.PositiveActivity + mc.Temp_Activity.NegativeActivity;
                        if (activity == activityMax)
                        {
                            bitmap.SetPixel(mcx, mcy, Color.White);
                        }
                        else
                        {
                            int brightness = (int)(255 * (activity - activityMin) / (activityMax - activityMin));
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

    public static Bitmap GetBitmapFromMiniColums_Activity_Code(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
    {
        Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

        using (Graphics g = Graphics.FromImage(bitmap))
        {
            // Устанавливаем черный фон
            g.Clear(Color.Black);
        }

        var miniColumns = cortex.MiniColumns.Data.OrderByDescending(mc => mc.Temp_Activity).Take(7).ToArray();
        foreach (var mc in miniColumns)
        {
            bitmap.SetPixel(mc.MCX, mc.MCY, Color.White);
        }

        return bitmap;
    }

    public static Bitmap GetBitmapFromMiniColums_SuperActivityColor(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex, ActivitiyMaxInfo? activitiyMaxInfo)
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
                            bitmap.SetPixel(mcx, mcy, Color.White);
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
        //    foreach (Cortex.MiniColumn maxSuperActivityMiniColumn in activitiyMaxInfo.SuperActivityMax_Temp_MiniColumns)
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

    public static Bitmap GetBitmapFromMiniColums_SuperActivity_Code(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
    {
        Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

        using (Graphics g = Graphics.FromImage(bitmap))
        {
            // Устанавливаем черный фон
            g.Clear(Color.Black);
        }

        var miniColumns = cortex.MiniColumns.Data.OrderByDescending(mc => mc.Temp_SuperActivity).Take(7).ToArray();
        foreach (var mc in miniColumns)
        {
            bitmap.SetPixel(mc.MCX, mc.MCY, Color.White);
        }

        return bitmap;
    }

    public static Bitmap GetBitmapFromMiniColumsMemoriesColor(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
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
                if (mc is not null && mc.CortexMemories.Count > 0)
                {
                    Color color = GetAverageLABColor(mc.CortexMemories
                        .Where(cm => cm is not null && cm.DiscreteRandomVector_Color != Color.Black).Select(cm => cm!.DiscreteRandomVector_Color));

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

    public static Bitmap GetBitmapFromMiniColumsMemoriesCount(Ssz.AI.Models.AdvancedEmbeddingModel2.Cortex cortex)
    {
        Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

        int minCortexMemoriesCount = int.MaxValue;
        int maxCortexMemoriesCount = int.MinValue;

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null)
                {
                    if (mc.CortexMemories.Count > maxCortexMemoriesCount)
                        maxCortexMemoriesCount = mc.CortexMemories.Count;
                    if (mc.CortexMemories.Count < minCortexMemoriesCount)
                        minCortexMemoriesCount = mc.CortexMemories.Count;
                }
            }

        minCortexMemoriesCount = 0;

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null && maxCortexMemoriesCount != minCortexMemoriesCount)
                {
                    int brightness = (int)(255 * ((float)(mc.CortexMemories.Count - minCortexMemoriesCount)) / (maxCortexMemoriesCount - minCortexMemoriesCount));

                    bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, brightness, brightness));
                }
                else
                {
                    bitmap.SetPixel(mcx, mcy, Color.Black);
                }
            }

        return bitmap;
    }

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

    public static Bitmap GetBitmap(IEnumerable<Detector> activatedDetectors, int widthPixels, int heightPixels)
    {
        Bitmap bitmap = new Bitmap(widthPixels, heightPixels);

        for (int y = 0; y < heightPixels; y += 1)
        {
            for (int x = 0; x < widthPixels; x += 1)
            {
                bitmap.SetPixel(x, y, Color.Black);
            }
        }

        foreach (var detector in activatedDetectors)
        {
            bitmap.SetPixel((int)detector.CenterXPixels, (int)detector.CenterYPixels, Color.FromArgb(255, 200, 200, 200));
        }

        return bitmap;
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
            bitmap.SetPixel((int)(detector.CenterXPixels * 10.0), (int)(detector.CenterYPixels * 10.0), Color.FromArgb(255, 200, 200, 200));
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
        MatrixFloat_ColumnMajor? contextSyncingMatrixFloat,
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
        MatrixFloat_ColumnMajor? contextSyncingMatrixFloat,
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

    public static Bitmap GetMiniColumsActivityBitmap_Obsolete(Cortex_Simplified cortex, ActivitiyMaxInfo activitiyMaxInfo)
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
                Cortex_Simplified.MiniColumn mc = miniColumns[x, y];
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
                Cortex_Simplified.MiniColumn mc = miniColumns[x, y];
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

        foreach (Cortex_Simplified.MiniColumn? maxActivityMiniColumn in activitiyMaxInfo.ActivityMax_MiniColumns)
        {
            gradientBitmap.SetPixel(maxActivityMiniColumn.MCX, maxActivityMiniColumn.MCY, Color.Blue);
        }

        Cortex_Simplified.MiniColumn? maxSuperActivityMiniColumn = activitiyMaxInfo.GetSuperActivityMax_MiniColumn(new Random()) as Cortex_Simplified.MiniColumn;
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
    
    /// <summary>
    /// Преобразование HSV в RGB (используется для цветового кодирования угла градиента)
    /// </summary>
    /// <param name="hue">[0..1]</param>
    /// <param name="saturation">[0..1]</param>
    /// <param name="value">[0..1]</param>
    /// <returns></returns>
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
                if (mc is not null && !float.IsNaN(mc.Temp_Activity.Item1) && !float.IsNaN(mc.Temp_Activity.Item2))
                {
                    float activity = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
                    if (activity > activityMax)
                        activityMax = activity;
                    if (activity < activityMin)
                        activityMin = activity;
                }
            }

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null && !float.IsNaN(mc.Temp_Activity.Item1) && !float.IsNaN(mc.Temp_Activity.Item2))
                {
                    if (activityMax > activityMin)
                    {
                        float activity = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
                        if (activity == activityMax)
                        {
                            bitmap.SetPixel(mcx, mcy, Color.White);
                        }
                        else
                        {
                            int brightness = (int)(255 * (activity - activityMin) / (activityMax - activityMin));
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

    public static Bitmap GetBitmapFromMiniColums_ActivityColor(Cortex_Simplified cortex)
    {
        Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

        float activityMin = float.MaxValue;
        float activityMax = float.MinValue;

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null && !float.IsNaN(mc.Temp_Activity.Item1) && !float.IsNaN(mc.Temp_Activity.Item2))
                {
                    float activity = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
                    if (activity > activityMax)
                        activityMax = activity;
                    if (activity < activityMin)
                        activityMin = activity;
                }
            }

        foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
            {
                var mc = cortex.MiniColumns[mcx, mcy];
                if (mc is not null && !float.IsNaN(mc.Temp_Activity.Item1) && !float.IsNaN(mc.Temp_Activity.Item2))
                {
                    if (activityMax > activityMin)
                    {
                        float activity = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
                        if (activity == activityMax)                            
                        {
                            bitmap.SetPixel(mcx, mcy, Color.White);                                
                        }
                        else
                        {
                            int brightness = (int)(255 * (activity - activityMin) / (activityMax - activityMin));
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
                            bitmap.SetPixel(mcx, mcy, Color.White);
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

    //public static Bitmap GetBitmapFromMiniColums_SuperActivityColor(Cortex cortex, ActivitiyMaxInfo? activitiyMaxInfo)
    //{
    //    Bitmap bitmap = new Bitmap(cortex.MiniColumns.Dimensions[0], cortex.MiniColumns.Dimensions[1]);

    //    float superActivityMin = float.MaxValue;
    //    float superActivityMax = float.MinValue;

    //    foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
    //        foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
    //        {
    //            var mc = cortex.MiniColumns[mcx, mcy];
    //            if (mc is not null && !float.IsNaN(mc.Temp_SuperActivity))
    //            {
    //                if (mc.Temp_SuperActivity > superActivityMax)
    //                    superActivityMax = mc.Temp_SuperActivity;
    //                if (mc.Temp_SuperActivity < superActivityMin)
    //                    superActivityMin = mc.Temp_SuperActivity;
    //            }
    //        }

    //    foreach (int mcy in Enumerable.Range(0, cortex.MiniColumns.Dimensions[1]))
    //        foreach (int mcx in Enumerable.Range(0, cortex.MiniColumns.Dimensions[0]))
    //        {
    //            var mc = cortex.MiniColumns[mcx, mcy];
    //            if (mc is not null && !float.IsNaN(mc.Temp_SuperActivity))
    //            {
    //                if (superActivityMax > superActivityMin)
    //                {
    //                    if ((activitiyMaxInfo is not null &&
    //                            activitiyMaxInfo.SuperActivityMax_MiniColumns.Contains(mc)) ||
    //                            (activitiyMaxInfo is null &&
    //                            mc.Temp_SuperActivity == superActivityMax))
    //                    {
    //                        bitmap.SetPixel(mcx, mcy, Color.White);
    //                    }
    //                    else
    //                    {
    //                        int brightness = (int)(255 * (mc.Temp_SuperActivity - superActivityMin) / (superActivityMax - superActivityMin));
    //                        bitmap.SetPixel(mcx, mcy, Color.FromArgb(brightness, 0, 0));
    //                    }
    //                }
    //                else
    //                {
    //                    bitmap.SetPixel(mcx, mcy, Color.White);
    //                }
    //            }
    //            else
    //            {
    //                bitmap.SetPixel(mcx, mcy, Color.Black);
    //            }
    //        }

    //    //if (allMaxSuperActivity)
    //    //{
    //    //    foreach (Cortex.MiniColumn maxSuperActivityMiniColumn in activitiyMaxInfo.SuperActivityMax_MiniColumns)
    //    //    {
    //    //        bitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 255, 255));
    //    //    }
    //    //}
    //    //else
    //    //{
    //    //    Cortex.MiniColumn? maxSuperActivityMiniColumn = cortex.Temp_SuperActivityMax_MiniColumn;
    //    //    if (maxSuperActivityMiniColumn is not null)
    //    //        bitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 255, 255));
    //    //}

    //    return bitmap;
    //}

    public static Bitmap GetBitmapFromMiniColums_SuperActivityColor(Cortex_Simplified cortex, ActivitiyMaxInfo? activitiyMaxInfo)
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
                            bitmap.SetPixel(mcx, mcy, Color.White);
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

    public static Bitmap GetMiniColumsActivityMaxBitmap(Cortex_Simplified cortex, ActivitiyMaxInfo activitiyMaxInfo, bool allSuperActivity)
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

        foreach (Cortex_Simplified.MiniColumn maxActivityMiniColumn in activitiyMaxInfo.ActivityMax_MiniColumns)
        {
            gradientBitmap.SetPixel(maxActivityMiniColumn.MCX, maxActivityMiniColumn.MCY, Color.FromArgb(255, 255, 0));
        }

        if (allSuperActivity)
        {
            foreach (Cortex_Simplified.MiniColumn maxSuperActivityMiniColumn in activitiyMaxInfo.SuperActivityMax_MiniColumns)
            {
                gradientBitmap.SetPixel(maxSuperActivityMiniColumn.MCX, maxSuperActivityMiniColumn.MCY, Color.FromArgb(255, 0, 0));
            }
        }
        else
        {
            Cortex_Simplified.MiniColumn? maxSuperActivityMiniColumn = cortex.Temp_SuperActivityMax_MiniColumn;
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

    public static Bitmap GetBitmapFromMiniColumsMemoriesCount(Cortex_Simplified cortex)
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

    public static Bitmap GetBitmapFromMiniColumsMemoriesColor(Cortex_Simplified cortex)
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

    public static Bitmap GetBigMatrixFloatBitmap(MatrixFloat_ColumnMajor bigMatrixFloat, int j)
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

    /// <summary>
    ///     Bad result
    /// </summary>
    /// <param name="colors"></param>
    /// <returns></returns>
    public static Color GetAverageColor(IEnumerable<Color> colors)
    {
        // Суммируем компоненты R, G, B отдельно для параллелизма; используем long для избежания переполнения (max n*255 = ~2^31 для n<2^24).
        long sumR = 0;  // Сумма красных: Σ R_i, где R_i = color_i.R (0 ≤ R_i ≤ 255)
        long sumG = 0;  // Сумма зелёных: Σ G_i, где G_i = color_i.G (0 ≤ G_i ≤ 255)
        long sumB = 0;  // Сумма синих: Σ B_i, где B_i = color_i.B (0 ≤ B_i ≤ 255)

        int n = 0;  // Счётчик элементов: n += 1 для каждого цвета.

        // LINQ для сумм: итерация по colors, добавление к суммам. Average() = sum / n с double, но Sum() для long точнее.
        // Разбор: colors.Sum(c => (long)c.R) эквивалентно, но здесь ручной цикл для детального комментария и Span-поддержки.
        foreach (Color color in colors)
        {
            // Добавляем значения: sumR += c.R эквивалентно sumR = sumR + (long)color.R для предотвращения потери при суммировании байтов.
            sumR += color.R;
            sumG += color.G;
            sumB += color.B;
            n += 1;  // Увеличиваем счётчик на 1
        }

        // Вычисляем средние: avg_R = sumR / n в double для точности, затем Math.Round для байта (округление к ближайшему).
        // Обозначения: avg_R = round( (Σ R_i) / n ), где round(x) = floor(x + 0.5) для целых.
        // Clamp не нужен, т.к. 0 ≤ avg_c ≤ 255 по свойствам суммы.
        int avgR = (int)Math.Round((double)sumR / n);  // Средний красный: формула avg_R с округлением.
        int avgG = (int)Math.Round((double)sumG / n);  // Средний зелёный: аналогично.
        int avgB = (int)Math.Round((double)sumB / n);  // Средний синий: аналогично.

        // Создаём новый Color без альфы (A=255 по умолчанию); для альфы: sumA / n и FromArgb(avgA, avgR, avgG, avgB).
        // FromArgb(int, int, int): конструктор с R, G, B; гарантирует валидные байты.
        return Color.FromArgb(avgR, avgG, avgB);
    }

    /// <summary>
    /// Метод-расширение для перцептивно корректного усреднения цветов через квадратичное среднее (RMS).
    /// Этот подход компенсирует гамма-коррекцию sRGB, давая более яркие и визуально точные результаты.
    /// </summary>
    /// <param name="colors">Коллекция объектов Color для смешения. n = colors.Count() — количество цветов.</param>
    /// <returns>Усредненный Color в sRGB: avg = Color.FromArgb(round(avg_R_sRGB), round(avg_G_sRGB), round(avg_B_sRGB)),
    /// где avg_c_sRGB = sqrt( (1/n) * Σ c_i² ) для компоненты c (R, G или B).</returns>
    /// <remarks>
    /// Формула квадратичного среднего (RMS, root mean square):
    /// avg_c = sqrt( (1/n) * Σ_{i=1}^n (c_i)² ),
    /// где c_i — значение компоненты i-го цвета (byte в [0, 255]),
    /// n — количество цветов (|colors| > 0).
    /// 
    /// Обоснование: sRGB использует гамма-кодирование V_sRGB ≈ V_linear^(1/2.2) для соответствия
    /// нелинейному восприятию яркости человеком. Усреднение закодированных значений эквивалентно
    /// геометрическому среднему интенсивности света, что визуально темнее арифметического среднего.
    /// Квадратичное среднее (RMS) аппроксимирует декодирование гаммы (γ≈2), усреднение в линейном
    /// пространстве и обратное кодирование, давая перцептивно корректный результат без сложных преобразований.
    /// 
    /// Производительность: O(n) время, O(1) память. Быстрее LAB-методов в ~10× раз (только sqrt и умножения).
    /// Точность: визуально близко к LAB для большинства случаев, идеально для UI/графики.
    /// </remarks>
    public static Color GetAveragePerceptualColor(IEnumerable<Color> colors)
    {
        if (colors == null || !colors.Any())
        {
            throw new ArgumentException("Коллекция цветов не может быть null или пустой.", nameof(colors));
        }

        // Суммы квадратов компонент для компенсации гаммы sRGB.
        // Используем double для избежания переполнения: max = n * 255² ≈ n * 65025 < 2^53 для n < 10^9.
        double sumRSquared = 0.0;  // Сумма квадратов R: Σ R_i², где R_i = color_i.R (0 ≤ R_i ≤ 255)
        double sumGSquared = 0.0;  // Сумма квадратов G: Σ G_i², где G_i = color_i.G
        double sumBSquared = 0.0;  // Сумма квадратов B: Σ B_i², где B_i = color_i.B

        int n = 0;  // Счётчик цветов: n += 1 для каждого элемента (избегаем ++ согласно требованиям).

        // Итерируем по коллекции, суммируя квадраты компонент.
        foreach (Color color in colors)
        {
            // Квадраты компонент: (R_i)² для моделирования линейной интенсивности света.
            // Обозначения: R_i² = R_i * R_i, где умножение даёт физическую энергию (∝ фотоны).
            int r = color.R;  // Извлекаем красную компоненту (byte 0-255).
            int g = color.G;  // Извлекаем зелёную компоненту.
            int b = color.B;  // Извлекаем синюю компоненту.

            // Добавляем квадраты к суммам: sumR² += R_i² без приведения к double (int * int = int, затем implicit cast).
            sumRSquared += r * r;
            sumGSquared += g * g;
            sumBSquared += b * b;

            n += 1;  // Увеличиваем счётчик на 1 (вместо n++).
        }

        // Вычисляем квадратичные средние: avg_c = sqrt( sumC² / n ) для каждой компоненты c.
        // Формула RMS: sqrt( (1/n) * Σ c_i² ) = sqrt( mean(c²) ), где mean(x) = Σx / n.
        // Math.Sqrt в .NET 9 JIT-компилируется в аппаратную инструкцию VSQRTSD (AVX), O(1) производительность.
        double avgR = Math.Sqrt(sumRSquared / n);  // Среднее R: sqrt( (Σ R_i²) / n )
        double avgG = Math.Sqrt(sumGSquared / n);  // Среднее G: аналогично.
        double avgB = Math.Sqrt(sumBSquared / n);  // Среднее B: аналогично.

        // Округляем до byte (0-255): Math.Round использует банковское округление (к ближайшему чётному).
        // Clamp не требуется: 0 ≤ sqrt(x) ≤ 255 для x ≤ 255², гарантировано по математике.
        int finalR = (int)Math.Round(avgR);  // Округление avg_R до целого.
        int finalG = (int)Math.Round(avgG);  // Округление avg_G.
        int finalB = (int)Math.Round(avgB);  // Округление avg_B.

        // Создаём результирующий Color без альфы (A = 255 по умолчанию, непрозрачный).
        // FromArgb(int, int, int): конструктор с RGB-компонентами, автоматически валидирует диапазон [0, 255].
        return Color.FromArgb(finalR, finalG, finalB);
    }

    /// <summary>
    /// Усреднение цветов в перцептивно-однородном пространстве LAB (CIELAB) для максимальной точности.
    /// Используйте, когда визуальная точность критична (например, цветокоррекция, палитры для дизайна).
    /// </summary>
    /// <param name="colors">Коллекция объектов Color для смешения в LAB-пространстве.</param>
    /// <returns>Средний Color, усредненный в CIELAB и преобразованный обратно в sRGB.</returns>
    /// <remarks>
    /// Пространство LAB (CIE L*a*b*):
    /// - L* = Lightness (яркость): 0 (чёрный) ≤ L* ≤ 100 (белый).
    /// - a* = Green↔Red axis: -128 (зелёный) ≤ a* ≤ 127 (красный).
    /// - b* = Blue↔Yellow axis: -128 (синий) ≤ b* ≤ 127 (жёлтый).
    /// 
    /// Преобразование sRGB → LAB:
    /// 1. Нормализация RGB: R' = R / 255, G' = G / 255, B' = B / 255.
    /// 2. Декодирование гаммы sRGB (γ = 2.4 для sRGB):
    ///    R_linear = (R' > 0.04045) ? ((R' + 0.055) / 1.055)^2.4 : R' / 12.92, аналогично для G, B.
    /// 3. Преобразование в XYZ (D65 illuminant, 2° observer):
    ///    X = 0.4124564*R_lin + 0.3575761*G_lin + 0.1804375*B_lin,
    ///    Y = 0.2126729*R_lin + 0.7151522*G_lin + 0.0721750*B_lin,
    ///    Z = 0.0193339*R_lin + 0.1191920*G_lin + 0.9503041*B_lin.
    /// 4. Нормализация к белой точке D65: X_n = 95.047, Y_n = 100.000, Z_n = 108.883.
    /// 5. Функция f(t) для LAB: f(t) = (t > δ³) ? t^(1/3) : (t / (3δ²)) + (4/29), где δ = 6/29.
    /// 6. Вычисление LAB:
    ///    L* = 116 * f(Y/Y_n) - 16,
    ///    a* = 500 * (f(X/X_n) - f(Y/Y_n)),
    ///    b* = 200 * (f(Y/Y_n) - f(Z/Z_n)).
    /// 
    /// Обратное преобразование LAB → sRGB: инверсия шагов 6→5→4→3→2→1.
    /// Производительность: O(n) время, но ~10× медленнее RMS из-за pow/sqrt вызовов и матричных умножений.
    /// Точность: перцептивно идеальная, особенно для контрастных цветов (Delta E > 4).
    /// </remarks>
    public static Color GetAverageLABColor(IEnumerable<Color> colors)
    {
        // Референсная белая точка D65 (2° observer, стандарт для sRGB):
        // Обозначения: X_n, Y_n, Z_n — XYZ-координаты идеального белого (illuminant D65).
        const double Xn = 95.047;   // X-компонента белой точки D65.
        const double Yn = 100.000;  // Y-компонента (яркость нормализована к 100).
        const double Zn = 108.883;  // Z-компонента.

        // Пороговое значение для функции f(t) в LAB: δ = 6/29 ≈ 0.206897.
        // δ³ = (6/29)³ ≈ 0.008856: если t > δ³, используем кубический корень, иначе линейную аппроксимацию.
        const double delta = 6.0 / 29.0;
        const double deltaCubed = delta * delta * delta;  // δ³ для сравнения.

        double sumL = 0.0;  // Сумма L* (яркость): Σ L_i*, где L_i* из LAB i-го цвета.
        double sumA = 0.0;  // Сумма a* (green↔red): Σ a_i*.
        double sumB = 0.0;  // Сумма b* (blue↔yellow): Σ b_i*.
        int n = 0;

        bool any = false;

        // Конвертируем каждый цвет в LAB и суммируем компоненты.
        foreach (Color color in colors)
        {
            any = true;
            // Шаг 1: Нормализация RGB к [0, 1].
            // Обозначения: R', G', B' — нормализованные значения (double в [0, 1]).
            double r01 = color.R / 255.0;
            double g01 = color.G / 255.0;
            double b01 = color.B / 255.0;

            // Шаг 2: Декодирование гаммы sRGB → линейный RGB.
            // Формула: если V > 0.04045, то V_lin = ((V + 0.055) / 1.055)^2.4, иначе V_lin = V / 12.92.
            // Обозначения: R_linear, G_linear, B_linear — линейные интенсивности света (double в [0, 1]).
            double rLin = (r01 > 0.04045) ? Math.Pow((r01 + 0.055) / 1.055, 2.4) : r01 / 12.92;
            double gLin = (g01 > 0.04045) ? Math.Pow((g01 + 0.055) / 1.055, 2.4) : g01 / 12.92;
            double bLin = (b01 > 0.04045) ? Math.Pow((b01 + 0.055) / 1.055, 2.4) : b01 / 12.92;

            // Шаг 3: Преобразование линейного RGB → XYZ (матрица для sRGB D65).
            // Формула: X = 0.4124564*R_lin + 0.3575761*G_lin + 0.1804375*B_lin, аналогично для Y, Z.
            // Обозначения: X, Y, Z — координаты в CIE XYZ (Y ≈ яркость, X и Z — цветность).
            double x = 0.4124564 * rLin + 0.3575761 * gLin + 0.1804375 * bLin;
            double y = 0.2126729 * rLin + 0.7151522 * gLin + 0.0721750 * bLin;
            double z = 0.0193339 * rLin + 0.1191920 * gLin + 0.9503041 * bLin;

            // Шаг 4: Нормализация XYZ к белой точке D65.
            // Обозначения: xNorm = X / X_n, yNorm = Y / Y_n, zNorm = Z / Z_n (double, безразмерные).
            double xNorm = x / Xn;
            double yNorm = y / Yn;
            double zNorm = z / Zn;

            // Шаг 5: Функция f(t) для LAB.
            // Формула: f(t) = t^(1/3) если t > δ³, иначе f(t) = t / (3δ²) + 4/29.
            // Обозначения: fx, fy, fz — преобразованные значения для L*a*b*.
            double fx = (xNorm > deltaCubed) ? Math.Pow(xNorm, 1.0 / 3.0) : (xNorm / (3.0 * delta * delta) + 4.0 / 29.0);
            double fy = (yNorm > deltaCubed) ? Math.Pow(yNorm, 1.0 / 3.0) : (yNorm / (3.0 * delta * delta) + 4.0 / 29.0);
            double fz = (zNorm > deltaCubed) ? Math.Pow(zNorm, 1.0 / 3.0) : (zNorm / (3.0 * delta * delta) + 4.0 / 29.0);

            // Шаг 6: Вычисление L*a*b*.
            // Формулы: L* = 116*fy - 16, a* = 500*(fx - fy), b* = 200*(fy - fz).
            // Обозначения: lStar (L*) в [0, 100], aStar (a*) в [-128, 127], bStar (b*) в [-128, 127].
            double lStar = 116.0 * fy - 16.0;
            double aStar = 500.0 * (fx - fy);
            double bStar = 200.0 * (fy - fz);

            // Суммируем LAB-компоненты.
            sumL += lStar;
            sumA += aStar;
            sumB += bStar;
            n += 1;
        }

        if (!any)
            return Color.Black;

        // Вычисляем средние LAB: avg_L* = Σ L_i* / n, аналогично для a*, b*.
        double avgL = sumL / n;
        double avgA = sumA / n;
        double avgB = sumB / n;

        // Обратное преобразование LAB → sRGB.
        // Шаг 6' (инверсия): Из L*a*b* → fx, fy, fz.
        // Формула: fy = (L* + 16) / 116, fx = a*/500 + fy, fz = fy - b*/200.
        double fy2 = (avgL + 16.0) / 116.0;
        double fx2 = avgA / 500.0 + fy2;
        double fz2 = fy2 - avgB / 200.0;

        // Шаг 5' (инверсия): Из fx, fy, fz → X/X_n, Y/Y_n, Z/Z_n.
        // Формула инверсии f⁻¹(t): если t > δ, то f⁻¹(t) = t³, иначе f⁻¹(t) = 3δ²*(t - 4/29).
        double xNorm2 = (fx2 > delta) ? (fx2 * fx2 * fx2) : (3.0 * delta * delta * (fx2 - 4.0 / 29.0));
        double yNorm2 = (fy2 > delta) ? (fy2 * fy2 * fy2) : (3.0 * delta * delta * (fy2 - 4.0 / 29.0));
        double zNorm2 = (fz2 > delta) ? (fz2 * fz2 * fz2) : (3.0 * delta * delta * (fz2 - 4.0 / 29.0));

        // Шаг 4' (инверсия): Денормализация к абсолютным XYZ.
        double x2 = xNorm2 * Xn;
        double y2 = yNorm2 * Yn;
        double z2 = zNorm2 * Zn;

        // Шаг 3' (инверсия): XYZ → линейный RGB (обратная матрица).
        // Обратная матрица для sRGB D65:
        double rLin2 = 3.2404542 * x2 - 1.5371385 * y2 - 0.4985314 * z2;
        double gLin2 = -0.9692660 * x2 + 1.8760108 * y2 + 0.0415560 * z2;
        double bLin2 = 0.0556434 * x2 - 0.2040259 * y2 + 1.0572252 * z2;

        // Clamp линейных значений к [0, 1] для обработки выхода за gamut sRGB.
        // Обозначения: Math.Max(0, Math.Min(1, x)) ограничивает x в [0, 1].
        rLin2 = Math.Max(0.0, Math.Min(1.0, rLin2));
        gLin2 = Math.Max(0.0, Math.Min(1.0, gLin2));
        bLin2 = Math.Max(0.0, Math.Min(1.0, bLin2));

        // Шаг 2' (инверсия): Кодирование гаммы линейный RGB → sRGB.
        // Формула: если V_lin > 0.0031308, то V_sRGB = 1.055*V_lin^(1/2.4) - 0.055, иначе V_sRGB = 12.92*V_lin.
        double r01_2 = (rLin2 > 0.0031308) ? (1.055 * Math.Pow(rLin2, 1.0 / 2.4) - 0.055) : (12.92 * rLin2);
        double g01_2 = (gLin2 > 0.0031308) ? (1.055 * Math.Pow(gLin2, 1.0 / 2.4) - 0.055) : (12.92 * gLin2);
        double b01_2 = (bLin2 > 0.0031308) ? (1.055 * Math.Pow(bLin2, 1.0 / 2.4) - 0.055) : (12.92 * bLin2);

        // Шаг 1' (инверсия): Денормализация к [0, 255].
        int finalR = (int)Math.Round(r01_2 * 255.0);
        int finalG = (int)Math.Round(g01_2 * 255.0);
        int finalB = (int)Math.Round(b01_2 * 255.0);

        // Дополнительный clamp на случай ошибок округления.
        finalR = Math.Max(0, Math.Min(255, finalR));
        finalG = Math.Max(0, Math.Min(255, finalG));
        finalB = Math.Max(0, Math.Min(255, finalB));

        return Color.FromArgb(finalR, finalG, finalB);
    }

    public static Bitmap GetBitmapFromMiniColumsMemoriesColor(CortexVisualisationModel.Cortex cortex)
    {
        float miniColumnRadius_Pixels = 5.0f;
        float radius_Pixels = (cortex.Constants.HypercolumnDefinedRadius_MiniColumns + 1) * miniColumnRadius_Pixels * 2.0f;
        Bitmap bitmap = new Bitmap((int)(radius_Pixels * 2), (int)(radius_Pixels * 2));

        using (Graphics g = Graphics.FromImage(bitmap))
        {
            // Устанавливаем черный фон
            g.Clear(Color.Black);

            for (int mci = 0; mci < cortex.MiniColumns.Count; mci += 1)
            {
                var miniColumn = cortex.MiniColumns[mci];
                if (miniColumn.CortexMemories.Count > 0)
                {
                    Color color = GetAverageLABColor(miniColumn.CortexMemories
                        .Where(cm => cm is not null)
                        .Select(cm => cortex.InputItems[cm!.InputItemIndex])
                        .Where(ii => ii.Color != Color.Black)
                        .Select(ii => ii.Color));

                    g.FillEllipse(
                        new SolidBrush(color),
                        miniColumn.MCX * miniColumnRadius_Pixels * 2 + radius_Pixels - miniColumnRadius_Pixels - 1.0f,
                        miniColumn.MCY * miniColumnRadius_Pixels * 2 + radius_Pixels - miniColumnRadius_Pixels - 1.0f,
                        miniColumnRadius_Pixels * 2 + 2.0f,
                        miniColumnRadius_Pixels * 2 + 2.0f
                        );
                }
            }
        }

        return bitmap;
    }

    public static Image GetBitmapFromMiniColumsValue(CortexVisualisationModel.Cortex cortex, Func<CortexVisualisationModel.Cortex.MiniColumn, double> getValue, double? valueMin = null, double? valueMax = null)
    {
        double valueMin_Final;
        double valueMax_Final;
        if (valueMin is not null && valueMax is not null)
        {
            valueMin_Final = valueMin.Value;
            valueMax_Final = valueMax.Value;
        }
        else
        {
            double valueMin_Local = Double.MaxValue;
            double valueMax_Local = Double.MinValue;

            for (int mci = 0; mci < cortex.MiniColumns.Count; mci += 1)
            {
                var miniColumn = cortex.MiniColumns[mci];
                if (miniColumn.CortexMemories.Count > 0)
                {
                    double value = getValue(miniColumn);
                    if (value < valueMin_Local)
                        valueMin_Local = value;
                    if (value > valueMax_Local)
                        valueMax_Local = value;
                }
            }

            if (valueMin is not null)
                valueMin_Final = valueMin.Value;            
            else
                valueMin_Final = valueMin_Local;

            if (valueMax is not null)
                valueMax_Final = valueMax.Value;
            else
                valueMax_Final = valueMax_Local;
        }

        float miniColumnRadius_Pixels = 5.0f;
        float radius_Pixels = (cortex.Constants.HypercolumnDefinedRadius_MiniColumns + 1) * miniColumnRadius_Pixels * 2.0f;
        Bitmap bitmap = new Bitmap((int)(radius_Pixels * 2), (int)(radius_Pixels * 2));

        using (Graphics g = Graphics.FromImage(bitmap))
        {
            // Устанавливаем черный фон
            g.Clear(Color.Black);

            for (int mci = 0; mci < cortex.MiniColumns.Count; mci += 1)
            {
                var miniColumn = cortex.MiniColumns[mci];
                double value = getValue(miniColumn);
                int v = (int)(255 * (value - valueMin_Final) / (valueMax_Final - valueMin_Final));
                if (v > 255)
                    v = 255;
                else if (v < 0)
                    v = 0;

                Color color = Color.FromArgb(v, v, v);                

                g.FillEllipse(
                    new SolidBrush(color),
                    miniColumn.MCX * miniColumnRadius_Pixels * 2 + radius_Pixels - miniColumnRadius_Pixels - 1.0f,
                    miniColumn.MCY * miniColumnRadius_Pixels * 2 + radius_Pixels - miniColumnRadius_Pixels - 1.0f,
                    miniColumnRadius_Pixels * 2 + 2.0f,
                    miniColumnRadius_Pixels * 2 + 2.0f
                    );                
            }
        }

        return bitmap;
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
