using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.DrawingCore.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model5
    {
        #region construction and destruction

        /// <summary>
        ///     Построение "вертушки"
        /// </summary>
        public Model5()
        {
            string labelsPath = @"Data\train-labels.idx1-ubyte"; // Укажите путь к файлу меток
            string imagesPath = @"Data\train-images.idx3-ubyte"; // Укажите путь к файлу изображений

            var (labels, images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            List<GradientInPoint[,]> gradientMatricesCollection = new(images.Length);
            foreach (int i in Enumerable.Range(0, images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                gradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            Retina = new Retina(Constants, gradientDistribution, Constants.AngleRangesCount, Constants.MagnitudeRangesCount, Constants.HashLength);

            Cortex = new Cortex(Constants, Retina);

            // Прогон всех картинок
            int gmi = 0;
            foreach (var gradientMatrix in gradientMatricesCollection.Take(2000))
            {
                SuperActivitiyMaxInfo finalSuperActivitiyMaxInfo = GetFinalSuperActivitiyMaxInfo(gradientMatrix);

                // Сохраняем воспоминание в миниколонке-победителе.
                var miniColumn = finalSuperActivitiyMaxInfo.MiniColumn;
                if (miniColumn is not null)
                {
                    miniColumn.Temp_Memories.Add(miniColumn.Temp_Hash);
                }

                gmi += 1;
            }
        }        

        #endregion

        #region public functions

        public readonly ModelConstants Constants = new();

        public readonly Retina Retina;

        public readonly Cortex Cortex; 

        public int CenterX { get; set; }
        public int CenterXDelta { get; set; }
        public int CenterY { get; set; }
        public double AngleDelta { get; set; }
        public double Angle { get; set; }

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
            Bitmap resizedBitmap = new Bitmap(MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
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
                g.DrawImage(originalBitmap, new Rectangle(0, 0, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight), new Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height), GraphicsUnit.Pixel);
            }

            // Применяем оператор Собеля к первому изображению
            GradientInPoint[,] gradientMatrix = SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);           
                    

            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);

            SuperActivitiyMaxInfo finalSuperActivitiyMaxInfo = GetFinalSuperActivitiyMaxInfo(gradientMatrix);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.GetLength(0) * Retina.Detectors.GetLength(1));
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.GetLength(1)))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.GetLength(0)))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex);

            return [resizedBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }

        #endregion        

        private SuperActivitiyMaxInfo GetFinalSuperActivitiyMaxInfo(GradientInPoint[,] gradientMatrix)
        {
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    i =>
                    {
                        var d = Cortex.SubArea_Detectors[i];
                        d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                    });

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                i =>
                {
                    var mc = Cortex.SubArea_MiniColumns[i];
                    mc.Temp_Activity = mc.GetActivity();
                });

            SuperActivitiyMaxInfo finalSuperActivitiyMaxInfo = new();
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_MiniColumns.Length,
                    () => new SuperActivitiyMaxInfo(), // method to initialize the local variable
                    (i, loopState, superActivitiyMaxInfo) => // method invoked by the loop on each iteration
                    {
                        var mc = Cortex.SubArea_MiniColumns[i];
                        mc.Temp_SuperActivity = mc.GetSuperActivity();

                        if (mc.Temp_SuperActivity > superActivitiyMaxInfo.SuperActivity)
                        {
                            superActivitiyMaxInfo.SuperActivity = mc.Temp_SuperActivity;
                            superActivitiyMaxInfo.MiniColumn = mc;
                        }

                        return superActivitiyMaxInfo; // value to be passed to next iteration
                    },
                    superActivitiyMaxInfo => // Method to be executed when each partition has completed.
                    {
                        lock (finalSuperActivitiyMaxInfo)
                        {
                            if (superActivitiyMaxInfo.SuperActivity > finalSuperActivitiyMaxInfo.SuperActivity)
                            {
                                finalSuperActivitiyMaxInfo.SuperActivity = superActivitiyMaxInfo.SuperActivity;
                                finalSuperActivitiyMaxInfo.MiniColumn = superActivitiyMaxInfo.MiniColumn;
                            }
                        }
                    });

            return finalSuperActivitiyMaxInfo;
        }

        private class SuperActivitiyMaxInfo
        {
            public MiniColumn? MiniColumn = null;
            public float SuperActivity = float.MinValue;
        }

        /// <summary>        
        ///     Константы данной модели
        /// </summary>
        public class ModelConstants : ICortexConstants
        {
            /// <summary>
            ///     Ширина основного изображения
            /// </summary>
            public int ImageWidth => MNISTHelper.MNISTImageWidth;

            /// <summary>
            ///     Высота основного изображения
            /// </summary>
            public int ImageHeight => MNISTHelper.MNISTImageHeight;

            public int AngleRangesCount => 4;

            public int MagnitudeRangesCount => 4;

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

            /// <summary>
            ///     Расстояние между детекторами по коризонтали и вертикали  
            /// </summary>
            public double DetectorDelta => 0.1;

            /// <summary>
            ///     Количество детекторов, видимых одной миниколонкой
            /// </summary>
            public int MiniColumnVisibleDetectorsCount => 250;            

            public int HashLength => 200;

            /// <summary>
            ///     Количество миниколонок в подобласти
            /// </summary>
            public int? SubAreaMiniColumnsCount => 10000;

            /// <summary>
            ///     Индекс X центра подобласти
            /// </summary>
            public int SubAreaCenter_Cx => 100;

            /// <summary>
            ///     Индекс Y центра подобласти
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
            public int NearestMiniColumnsDelta => 5;

            public double NearestMiniColumnsK => 5;
        }        
    }
}
