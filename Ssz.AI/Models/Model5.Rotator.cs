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

            (Labels, Images) = MNISTHelper.ReadMNIST(labelsPath, imagesPath);

            GradientDistribution gradientDistribution = new();

            GradientMatricesCollection = new(Images.Length);
            foreach (int i in Enumerable.Range(0, Images.Length))
            {
                // Применяем оператор Собеля
                GradientInPoint[,] gm = SobelOperator.ApplySobel(Images[i], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);
                GradientMatricesCollection.Add(gm);
                SobelOperator.CalculateDistribution(gm, gradientDistribution);
            }

            Retina = new Retina(Constants, gradientDistribution, Constants.AngleRangesCount, Constants.MagnitudeRangesCount, Constants.HashLength);

            Cortex = new Cortex(Constants, Retina);            
            
            CurrentMnistImageIndex = -1; // Перед первым элементом

            // Прогон картинок
            DoSteps(10000);
        }        

        #endregion

        #region public functions

        public readonly ModelConstants Constants = new();

        public readonly byte[] Labels;
        public readonly byte[][] Images;
        public readonly List<GradientInPoint[,]> GradientMatricesCollection;
        public int CurrentMnistImageIndex = 0;

        public readonly Retina Retina;

        public readonly Cortex Cortex;        

        public int Generated_CenterX { get; set; }
        public int Generated_CenterXDelta { get; set; }
        public int Generated_CenterY { get; set; }
        public double Generated_AngleDelta { get; set; }
        public double Generated_Angle { get; set; }        

        public Image[] GetImages1(double positionK, double angleK)
        {
            // Создаем изображение размером 280x280           

            Generated_CenterXDelta = (int)(positionK * Constants.GeneratedImageWidth / 2.0);
            Generated_CenterX = (int)(Constants.GeneratedImageWidth / 2.0) + Generated_CenterXDelta;
            Generated_CenterY = (int)(Constants.GeneratedImageHeight / 2.0);

            Generated_AngleDelta = angleK * 2.0 * Math.PI;
            Generated_Angle = Math.PI / 2 + Generated_AngleDelta;

            // Длина линии
            int lineLength = 100;

            // Рассчитываем конечные координаты линии
            int endX = (int)(Generated_CenterX + lineLength * Math.Cos(Generated_Angle));
            int endY = (int)(Generated_CenterY + lineLength * Math.Sin(Generated_Angle));

            // Рассчитываем начальные координаты линии (в противоположном направлении)
            int startX = (int)(Generated_CenterX - lineLength * Math.Cos(Generated_Angle));
            int startY = (int)(Generated_CenterY - lineLength * Math.Sin(Generated_Angle));

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

            ActivitiyMaxInfo activitiyMaxInfo = new();
                
            GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.GetLength(0) * Retina.Detectors.GetLength(1));
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.GetLength(1)))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.GetLength(0)))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            return [resizedBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }

        public Image[] GetImages2(int stepsCount)
        {
            DoSteps(stepsCount);

            var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

            var gradientBitmap = Visualisation.GetBitmap(gradientMatrix);

            ActivitiyMaxInfo activitiyMaxInfo = new();

            GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.GetLength(0) * Retina.Detectors.GetLength(1));
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.GetLength(1)))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.GetLength(0)))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            var originalBitmap = MNISTHelper.GetBitmap(Images[CurrentMnistImageIndex], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            return [originalBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }

        public Image[] GetImages3()
        {
            //Random random = new();
            //var hash0 = new float[Constants.HashLength];
            //foreach (var _ in Enumerable.Range(0, Constants.InitialMemoryBitsCount))
            //{
            //    hash0[random.Next(hash0.Length)] = 1.0f;
            //}            

            int currentMnistImageIndex = 0;
            var centerMiniColumn_Hash = new float[Constants.HashLength];
            foreach (var _ in Enumerable.Range(0, 10000))
            {
                currentMnistImageIndex += 1;

                var gradientMatrix = GradientMatricesCollection[currentMnistImageIndex];

                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                    });

                var centerMiniColumn = Cortex.CenterMiniColumn!;
                centerMiniColumn.CalculateHash(centerMiniColumn_Hash);

                if (TensorPrimitives.Sum(centerMiniColumn_Hash) < Constants.MinBitsInHashForMemory)
                    continue;

                bool found = false;

                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.VisualizationTableItems.Count,
                    (di, pls) =>
                    {
                        var visualizationTableItem = Cortex.VisualizationTableItems[di];
                        var cosineSimilarity = TensorPrimitives.CosineSimilarity(centerMiniColumn_Hash, visualizationTableItem.Hash);
                        if (cosineSimilarity > 0.9)
                        {
                            found = true;
                            pls.Stop();
                        }
                    });

                if (!found)
                {
                    var bitmap = Visualisation.GetBitmap(gradientMatrix);

                    var subBitmap = BitmapHelper.GetSubBitmap(
                        bitmap, 
                        (int)(centerMiniColumn.CenterX / Constants.DetectorDelta),
                        (int)(centerMiniColumn.CenterY / Constants.DetectorDelta),
                        Cortex.DetectorsVisibleRadius);
                    //Bitmap image = MNISTHelper.GetBitmap(Images[CurrentMnistImageIndex], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight, );

                    Color color;
                    if (Cortex.VisualizationTableItems.Count == 0)
                        color = Visualisation.ColorFromHSV(0, 1, 1);
                    else
                        color = Visualisation.ColorFromHSV(360 * TensorPrimitives.CosineSimilarity(Cortex.VisualizationTableItems[0].Hash, centerMiniColumn_Hash), 1, 1);

                    Cortex.VisualizationTableItems.Add(new VisualizationTableItem
                    {
                        Hash = (float[])centerMiniColumn_Hash.Clone(),
                        Color = color,
                        Image = subBitmap
                    });
                }
            }


            foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
            {
                MiniColumn mc = Cortex.SubArea_MiniColumns[mci];
                mc.Temp_Color = Color.Black;
            }

            ActivitiyMaxInfo activitiyMaxInfo = new();                        
            foreach (var vti in Enumerable.Range(0, Cortex.VisualizationTableItems.Count))
            {
                var visualizationTableItem = Cortex.VisualizationTableItems[vti];                

                GetSuperActivitiyMaxInfo(visualizationTableItem.Hash, activitiyMaxInfo);

                // Сохраняем воспоминание в миниколонке-победителе.
                MiniColumn? winnerMiniColumn = activitiyMaxInfo.SuperActivityMax_MiniColumn;
                if (winnerMiniColumn is not null)
                {
                    winnerMiniColumn.Temp_Color = visualizationTableItem.Color;
                }
            }


            var image = Visualisation.GetBitmapFromMiniColumsColor(Cortex);

            return [ image ];
        }

        #endregion        

        private void DoSteps(int stepsCount)
        {
            ActivitiyMaxInfo activitiyMaxInfo = new();
            MiniColumn? winnerMiniColumn;
            
            HashSet<MiniColumn> withMemoriesAdded_MiniColums = new();
            foreach (var _ in Enumerable.Range(0, stepsCount))
            {
                CurrentMnistImageIndex += 1;

                var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

                // Sleep and refresh all minicolumns
                if (CurrentMnistImageIndex > 0 && CurrentMnistImageIndex % 1000 == 0)
                {
                    foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
                    {
                        MiniColumn mc = Cortex.SubArea_MiniColumns[mci];

                        foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                        {
                            Memory memory = mc.Memories[mi];
                            if (memory.IsDeleted)
                                continue;

                            memory.IsDeleted = true;
                            mc.Memories[mi] = memory;

                            GetSuperActivitiyMaxInfo(memory.Hash, activitiyMaxInfo);

                            // Сохраняем воспоминание в миниколонке-победителе.
                            winnerMiniColumn = activitiyMaxInfo.SuperActivityMax_MiniColumn;
                            if (winnerMiniColumn is not null)
                            {
                                memory.IsDeleted = false;
                                if (ReferenceEquals(winnerMiniColumn, mc))
                                    mc.Memories[mi] = memory;
                                else
                                    winnerMiniColumn.Memories.Add(memory);
                            }
                        }
                    }

                    foreach (var mci in Enumerable.Range(0, Cortex.SubArea_MiniColumns.Length))
                    {
                        MiniColumn mc = Cortex.SubArea_MiniColumns[mci];

                        mc.Temp_Memories.Clear();

                        foreach (var mi in Enumerable.Range(0, mc.Memories.Count))
                        {
                            Memory memory = mc.Memories[mi];
                            if (memory.IsDeleted)
                                continue;

                            mc.Temp_Memories.Add(memory);
                        }

                        (mc.Memories, mc.Temp_Memories) = (mc.Temp_Memories, mc.Memories);
                    }
                }

                GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

                // Сохраняем воспоминание в миниколонке-победителе.
                winnerMiniColumn = activitiyMaxInfo.SuperActivityMax_MiniColumn;
                if (winnerMiniColumn is not null)
                {
                    if (TensorPrimitives.Sum(winnerMiniColumn.Temp_Hash) >= Constants.MinBitsInHashForMemory)
                    {
                        winnerMiniColumn.Memories.Add(new Memory { Hash = (float[])winnerMiniColumn.Temp_Hash.Clone() });
                        withMemoriesAdded_MiniColums.Add(winnerMiniColumn);
                    }
                    else
                    {
                        // Не должно быть
                    }
                }                
            }
        }

        private void GetSuperActivitiyMaxInfo(GradientInPoint[,] gradientMatrix, ActivitiyMaxInfo activitiyMaxInfo)
        {            
            activitiyMaxInfo.MaxSuperActivity = float.MinValue;
            activitiyMaxInfo.MaxActivity = float.MinValue;
            activitiyMaxInfo.SuperActivityMax_MiniColumn = null;
            activitiyMaxInfo.ActivityMax_MiniColumn = null;

            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_Detectors.Length,
                    di =>
                    {
                        var d = Cortex.SubArea_Detectors[di];
                        d.Temp_IsActivated = d.GetIsActivated(gradientMatrix);
                    });

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.CalculateHash(mc.Temp_Hash);
                    mc.Temp_Activity = mc.GetActivity(mc.Temp_Hash);
                });
            
            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_MiniColumns.Length,
                    () => new ActivitiyMaxInfo(), // method to initialize the local variable
                    (mci, loopState, localActivitiyMaxInfo) => // method invoked by the loop on each iteration
                    {
                        var mc = Cortex.SubArea_MiniColumns[mci];
                        mc.Temp_SuperActivity = mc.GetSuperActivity();

                        if (mc.Temp_Activity > localActivitiyMaxInfo.MaxActivity)
                        {
                            localActivitiyMaxInfo.MaxActivity = mc.Temp_Activity;
                            localActivitiyMaxInfo.ActivityMax_MiniColumn = mc;
                        }

                        if (mc.Temp_SuperActivity > localActivitiyMaxInfo.MaxSuperActivity)
                        {
                            localActivitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
                            localActivitiyMaxInfo.SuperActivityMax_MiniColumn = mc;
                        }

                        return localActivitiyMaxInfo; // value to be passed to next iteration
                    },
                    localActivitiyMaxInfo => // Method to be executed when each partition has completed.
                    {
                        lock (activitiyMaxInfo)
                        {
                            if (localActivitiyMaxInfo.MaxActivity > activitiyMaxInfo.MaxActivity)
                            {
                                activitiyMaxInfo.MaxActivity = localActivitiyMaxInfo.MaxActivity;
                                activitiyMaxInfo.ActivityMax_MiniColumn = localActivitiyMaxInfo.ActivityMax_MiniColumn;
                            }

                            if (localActivitiyMaxInfo.MaxSuperActivity > activitiyMaxInfo.MaxSuperActivity)
                            {
                                activitiyMaxInfo.MaxSuperActivity = localActivitiyMaxInfo.MaxSuperActivity;
                                activitiyMaxInfo.SuperActivityMax_MiniColumn = localActivitiyMaxInfo.SuperActivityMax_MiniColumn;
                            }
                        }
                    });
        }

        private void GetSuperActivitiyMaxInfo(float[] hash, ActivitiyMaxInfo activitiyMaxInfo)
        {
            activitiyMaxInfo.MaxSuperActivity = float.MinValue;
            activitiyMaxInfo.MaxActivity = float.MinValue;
            activitiyMaxInfo.SuperActivityMax_MiniColumn = null;
            activitiyMaxInfo.ActivityMax_MiniColumn = null;

            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.Temp_Activity = mc.GetActivity(hash);
                });

            Parallel.For(
                    fromInclusive: 0,
                    toExclusive: Cortex.SubArea_MiniColumns.Length,
                    () => new ActivitiyMaxInfo(), // method to initialize the local variable
                    (i, loopState, localActivitiyMaxInfo) => // method invoked by the loop on each iteration
                    {
                        var mc = Cortex.SubArea_MiniColumns[i];
                        mc.Temp_SuperActivity = mc.GetSuperActivity();

                        if (mc.Temp_Activity > localActivitiyMaxInfo.MaxActivity)
                        {
                            localActivitiyMaxInfo.MaxActivity = mc.Temp_Activity;
                            localActivitiyMaxInfo.ActivityMax_MiniColumn = mc;
                        }

                        if (mc.Temp_SuperActivity > localActivitiyMaxInfo.MaxSuperActivity)
                        {
                            localActivitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
                            localActivitiyMaxInfo.SuperActivityMax_MiniColumn = mc;
                        }

                        return localActivitiyMaxInfo; // value to be passed to next iteration
                    },
                    localActivitiyMaxInfo => // Method to be executed when each partition has completed.
                    {
                        lock (activitiyMaxInfo)
                        {
                            if (localActivitiyMaxInfo.MaxActivity > activitiyMaxInfo.MaxActivity)
                            {
                                activitiyMaxInfo.MaxActivity = localActivitiyMaxInfo.MaxActivity;
                                activitiyMaxInfo.ActivityMax_MiniColumn = localActivitiyMaxInfo.ActivityMax_MiniColumn;
                            }

                            if (localActivitiyMaxInfo.MaxSuperActivity > activitiyMaxInfo.MaxSuperActivity)
                            {
                                activitiyMaxInfo.MaxSuperActivity = localActivitiyMaxInfo.MaxSuperActivity;
                                activitiyMaxInfo.SuperActivityMax_MiniColumn = localActivitiyMaxInfo.SuperActivityMax_MiniColumn;
                            }
                        }
                    });
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

            public int AngleRangesCount => 6;

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
            public int? SubAreaMiniColumnsCount => 300;

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
            public int MinBitsInHashForMemory => 6;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int NearestMiniColumnsDelta => 5;            

            public float MiniColumnMinimumActivity => 0.66f;

            /// <summary>
            ///     Верхний предел количества воспоминаний (для кэширования)
            /// </summary>
            public int MemoriesMaxCount => 1000;
        }        
    }
}
