﻿using Avalonia.Layout;
using Microsoft.AspNetCore.DataProtection.XmlEncryption;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using static Ssz.AI.Models.Cortex_Simplified;
using Size = System.Drawing.Size;

namespace Ssz.AI.Models
{
    public class Model4
    {
        #region construction and destruction

        /// <summary>
        ///     Построение графика распределения величин градиентов
        /// </summary>
        public Model4()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            List<DenseMatrix<GradientInPoint>> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, 5000))
            {
                // Применяем оператор Собеля
                DenseMatrix<GradientInPoint> gm = SobelOperator.ApplySobel(images[i], MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
                gradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution, Constants);
            }

            Random random = new(1);

            //_retina = new Retina(Constants, gradientDistribution, Constants.AngleRangesCount, Constants.MagnitudeRangesCount, Constants.HashLength);
            Retina = new Retina(Constants);
            Retina.GenerateOwnedData(random, Constants, gradientDistribution);
            Retina.Prepare();

            Cortex = new Cortex_Simplified(Constants, Retina);            
            Cortex.GenerateOwnedData(Retina);
            //Helpers.SerializationHelper.LoadFromFileIfExists(@"autoencoder.bin", Cortex, "autoencoder");
            Cortex.Prepare();

            DetectorsActivationHash = new float[Constants.HashLength];
            // Вызываем для вычисления начального вектора активации детекторов
            GetImages(0.0, 0.0);
            DetectorsActivationHash0 = (float[])DetectorsActivationHash.Clone();

            //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            //foreach (var gradientMatrix in gradientMatricesCollection)
            //{
            //    Parallel.For(
            //        fromInclusive: 0,
            //        toExclusive: Retina.Detectors.Dimensions[1],
            //        dy =>
            //        {
            //            foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
            //            {
            //                var d = Retina.Detectors[dx, dy];
            //                d.Temp_IsActivated = d.GetIsActivated_Obsolete(gradientMatrix);
            //            }                            
            //        });               

            //    foreach (var miniColumn in Cortex.MiniColumns.Data)
            //    {
            //        miniColumn.GetHash(miniColumn.Temp_Hash);
            //        int bitsCountInHash = (int)TensorPrimitives.Sum(miniColumn.Temp_Hash);
            //        //dataToDisplayHolder.MiniColumsActivatedDetectorsCountDistribution[activatedDetectors.Intersect(miniColumn.Detectors).Count()] += 1;
            //        dataToDisplayHolder.MiniColumsBitsCountInHashDistribution[bitsCountInHash] += 1;
            //    }                
            //}
        }

        #endregion

        #region public functions       

        public readonly ModelConstants Constants = new();

        public float[] DetectorsActivationHash0 { get; set; }
        public float[] DetectorsActivationHash { get; set; }

        public int CenterX { get; set; }
        public int CenterXDelta { get; set; }
        public int CenterY { get; set; }
        public double AngleDelta { get; set; }
        public double Angle { get; set; }

        public Retina Retina;

        public Cortex_Simplified Cortex { get; }

        public Image[] GetImages(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            CenterXDelta = (int)(positionK * Constants.GeneratedImageWidth / 2.0); 
            CenterX = (int)(Constants.GeneratedImageWidth / 2.0) + CenterXDelta;
            CenterY = (int)(Constants.GeneratedImageHeight / 2.0);

            AngleDelta = angleK * 2.0 * Math.PI;
            Angle = Math.PI / 2 + AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(CenterX + lineLength * Math.Cos(Angle));
            int endY = (int)(CenterY + lineLength * Math.Sin(Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(CenterX - lineLength * Math.Cos(Angle));
            int startY = (int)(CenterY - lineLength * Math.Sin(Angle));

            Bitmap originalBitmap = new Bitmap(Constants.GeneratedImageWidth, Constants.GeneratedImageHeight);
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
            Bitmap resizedBitmap = new Bitmap(MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
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
                g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению
            DenseMatrix<GradientInPoint> gradientMatrix = SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.MNISTImageWidthPixels, MNISTHelper.MNISTImageHeightPixels);
            
            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.CalculateIsActivated(Retina, gradientMatrix, Constants);
                        if (d.Temp_IsActivated)
                        {                            
                            activatedDetectors.Add(d);
                        }
                    });

            Cortex.CenterMiniColumn!.GetHash(DetectorsActivationHash);

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            return [resizedBitmap, gradientBitmap, detectorsActivationBitmap];
        }

        #endregion        

        #region private fields        
        

        #endregion

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : IConstants
        {            
            public int AngleRangeDegree_LimitMagnitude { get; set; } = 300;

            public double DetectorMinGradientMagnitude => 5;

            public int GeneratedMinGradientMagnitude => 5;

            public int GeneratedMaxGradientMagnitude => 1200;

            public int AngleRangeDegreeMin { get; set; } = 120;

            public int AngleRangeDegreeMax { get; set; } = 120;

            public int MagnitudeRangesCount => 3;

            public int GeneratedImageWidth => 280;
            public int GeneratedImageHeight => 280;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси X
            /// </summary>
            public int CortexWidth => 200;

            /// <summary>
            ///     Количество миниколонок в зоне коры по оси Y
            /// </summary>
            public int CortexHeight => 200;

            public int RetinaImageWidthPixels => MNISTHelper.MNISTImageWidthPixels;

            public int RetinaImageHeightPixels => MNISTHelper.MNISTImageHeightPixels;

            /// <summary>
            ///     Расстояние между детекторами по горизонтали и вертикали              
            /// </summary>
            public float RetinaDetectorsDeltaPixels => 0.1f;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;

            public int HashLength => 200;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 400;

            /// <summary>
            ///     Индекс X центра подобласти [0..CortexWidth]
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти [0..CortexHeight]
            /// </summary>
            public int SubAreaCenter_Cy => 100;

            /// <summary>
            ///     Количество бит в хэше в первоначальном случайном воспоминании миниколонки.
            /// </summary>
            public int InitialMemoryBitsCount => 11;

            /// <summary>
            ///     Минимальное число бит в хэше, что бы быть сохраненным в память
            /// </summary>
            public int MinBitsInHashForMemory => 7;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public float MiniColumnsMaxDistance => 5;            

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;

            /// <summary>
            ///     Длина короткого хэш-вектора
            /// </summary>
            public int ShortHashLength => 50;

            /// <summary>
            ///     Количество бит в коротком хэш-векторе
            /// </summary>
            public int ShortHashBitsCount => 11;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public float MemoryClustersThreshold => 0.66f;

            public int Angle_SmallPoints_Count => 1000;

            public float Angle_SmallPoints_Radius => 0.003f;

            public int Angle_BigPoints_Count => 200;

            public float Angle_BigPoints_Radius => 0.015f;

            /// <summary>
            ///     Нулевой уровень косинусного расстояния
            /// </summary>
            public float K0 { get; set; }
            /// <summary>
            ///     Порог косинусного расстояния для учета 
            /// </summary>
            public float K1 { get; set; }
            /// <summary>
            ///     Косинусное расстояние для пустой колонки
            /// </summary>
            public float K2 { get; set; }

            /// <summary>
            ///     K значимости соседей
            /// </summary>
            public float[] K3 { get; set; } = null!;

            public float K4 { get; set; }

            public float K5 { get; set; }

            public bool SuperactivityThreshold { get; set; }

            public float[] PositiveK { get; set; } = [0.16f, 0.05f];

            public float[] NegativeK { get; set; } = [0.16f, 0.05f];
        }
    }
}
