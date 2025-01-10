using Avalonia.Layout;
using Microsoft.Extensions.DependencyInjection;
using OpenCvSharp;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Views;
using Ssz.Utils;
using System;
using System.Collections.Generic;
using System.DrawingCore;
using System.DrawingCore.Drawing2D;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using Ude.Core;
using static Ssz.AI.Models.Cortex;
using Size = System.DrawingCore.Size;

namespace Ssz.AI.Models
{
    public class Model6
    {
        #region construction and destruction

        /// <summary>
        ///     Гистограммы для миниколонок
        /// </summary>
        public Model6()
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

            Retina = new Retina(Constants);
            Retina.GenereateOwnedData(Constants, gradientDistribution);

            Cortex = new Cortex(Constants, Retina);            
            
            CurrentMnistImageIndex = -1; // Перед первым элементом

            // Прогон картинок
            DoSteps_MNIST(2000);
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
            (GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            MiniColumnsActivity.ActivitiyMaxInfo activitiyMaxInfo = new();
                
            //GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
                Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubAreaMiniColumnsRadius + 2);
            //var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            return [resizedBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }

        private (GradientInPoint[,], Bitmap) GetGeneratedLine_gradientMatrix(double positionK, double angleK)
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
            return (SobelOperator.ApplySobel(resizedBitmap, MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight), resizedBitmap);
        }

        public Image[] GetImages2()
        {
            //var totalMnistBitmap = GetMnistTotalBitmap();

            var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

            var gradientBitmap = Visualisation.GetGradientBigBitmap(gradientMatrix);

            MiniColumnsActivity.ActivitiyMaxInfo activitiyMaxInfo = new();

            //GetSuperActivitiyMaxInfo(gradientMatrix, activitiyMaxInfo);

            List<Detector> activatedDetectors = new List<Detector>(Retina.Detectors.Dimensions[0] * Retina.Detectors.Dimensions[1]);
            foreach (int dy in Enumerable.Range(0, Retina.Detectors.Dimensions[1]))
                foreach (int dx in Enumerable.Range(0, Retina.Detectors.Dimensions[0]))
                {
                    Detector d = Retina.Detectors[dx, dy];
                    if (d.Temp_IsActivated)
                        activatedDetectors.Add(d);
                }
            var detectorsActivationBitmap = Visualisation.GetBitmap(activatedDetectors);

            var miniColumsActivityBitmap = BitmapHelper.GetSubBitmap(
                Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo),
                Cortex.MiniColumns.Dimensions[0] / 2,
                Cortex.MiniColumns.Dimensions[1] / 2,
                Cortex.SubAreaMiniColumnsRadius + 2);
            //var miniColumsActivityBitmap = Visualisation.GetMiniColumsActivityBitmap(Cortex, activitiyMaxInfo);

            var originalBitmap = MNISTHelper.GetBitmap(Images[CurrentMnistImageIndex], MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight);

            return [originalBitmap, gradientBitmap, detectorsActivationBitmap, miniColumsActivityBitmap];
        }        

        public Image[] GetImages3()
        {
            var image = Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex);

            return [ image ];
        }

        public void DoSteps_MNIST(int stepsCount)
        {
            var random = new Random();

            DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();

            dataToDisplayHolder.MiniColumsBitsCountInHashDistribution2 = new ulong[Constants.CortexWidth, Constants.CortexHeight, Constants.HashLength];

            foreach (var _ in Enumerable.Range(0, stepsCount))
            {
                CurrentMnistImageIndex += 1;

                var gradientMatrix = GradientMatricesCollection[CurrentMnistImageIndex];

                DoStep(gradientMatrix, dataToDisplayHolder, random);
            }
        }

        public void DoStep_GeneratedLine(double positionK, double angleK)
        {
            var random = new Random();

            MiniColumnsActivity.ActivitiyMaxInfo activitiyMaxInfo = new();

            (GradientInPoint[,] gradientMatrix, var resizedBitmap) = GetGeneratedLine_gradientMatrix(positionK, angleK);
            
            //DoStep(gradientMatrix, activitiyMaxInfo, random);            
        }

        #endregion

        private void DoStep(GradientInPoint[,] gradientMatrix, DataToDisplayHolder dataToDisplayHolder, Random random)
        {
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
                    mc.GetHash(mc.Temp_Hash);

                    int bitsCountInHash = (int)TensorPrimitives.Sum(mc.Temp_Hash);
                    //dataToDisplayHolder.MiniColumsActivatedDetectorsCountDistribution[activatedDetectors.Intersect(miniColumn.Detectors).Count()] += 1;
                    dataToDisplayHolder.MiniColumsBitsCountInHashDistribution2[mc.MCX, mc.MCY, bitsCountInHash] += 1;

                    if (bitsCountInHash >= 11)
                    {
                        mc.Memories.Add(new Memory { Hash = (float[])mc.Temp_Hash.Clone() });
                    }
                });
        }        

        private void GetSuperActivitiyMaxInfo2(VisualizationTableItem visualizationTableItem, MiniColumnsActivity.ActivitiyMaxInfo activitiyMaxInfo)
        {
            Parallel.For(
                fromInclusive: 0,
                toExclusive: Cortex.SubArea_MiniColumns.Length,
                mci =>
                {
                    var mc = Cortex.SubArea_MiniColumns[mci];
                    mc.Temp_Activity = MiniColumnsActivity.GetActivity(mc, visualizationTableItem.SubArea_MiniColumns_Hashes[mci]);
                });

            GetSuperActivitiyMaxInfo(activitiyMaxInfo);
        }

        private void GetSuperActivitiyMaxInfo(MiniColumnsActivity.ActivitiyMaxInfo activitiyMaxInfo)
        {
            activitiyMaxInfo.MaxActivity = float.MinValue;
            activitiyMaxInfo.ActivityMax_MiniColumns.Clear();

            activitiyMaxInfo.MaxSuperActivity = float.MinValue;
            activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();            

            foreach (var mc in Cortex.SubArea_MiniColumns)
            {
                mc.Temp_SuperActivity = MiniColumnsActivity.GetSuperActivity(mc);

                float a = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
                if (a > activitiyMaxInfo.MaxActivity)
                {
                    activitiyMaxInfo.MaxActivity = a;
                    activitiyMaxInfo.ActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
                }
                else if (a == activitiyMaxInfo.MaxActivity)
                {                    
                    activitiyMaxInfo.ActivityMax_MiniColumns.Add(mc);
                }

                if (mc.Temp_SuperActivity > activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Clear();
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
                }
                else if (mc.Temp_SuperActivity == activitiyMaxInfo.MaxSuperActivity)
                {
                    activitiyMaxInfo.SuperActivityMax_MiniColumns.Add(mc);
                }
            }

            //Parallel.For(
            //        fromInclusive: 0,
            //        toExclusive: Cortex.SubArea_MiniColumns.Length,
            //        () => new ActivitiyMaxInfo(), // method to initialize the local variable
            //        (mci, loopState, localActivitiyMaxInfo) => // method invoked by the loop on each iteration
            //        {
            //            var mc = Cortex.SubArea_MiniColumns[mci];
            //            mc.Temp_SuperActivity = mc.GetSuperActivity();

            //            float a = mc.Temp_Activity.Item1 + mc.Temp_Activity.Item2;
            //            if (a > localActivitiyMaxInfo.MaxActivity)
            //            {
            //                localActivitiyMaxInfo.MaxActivity = a;
            //                localActivitiyMaxInfo.ActivityMax_MiniColumn = mc;
            //            }

            //            if (mc.Temp_SuperActivity > localActivitiyMaxInfo.MaxSuperActivity)
            //            {
            //                localActivitiyMaxInfo.MaxSuperActivity = mc.Temp_SuperActivity;
            //                localActivitiyMaxInfo.SuperActivityMax_MiniColumn = mc;
            //            }

            //            return localActivitiyMaxInfo; // value to be passed to next iteration
            //        },
            //        localActivitiyMaxInfo => // Method to be executed when each partition has completed.
            //        {
            //            lock (activitiyMaxInfo)
            //            {
            //                if (localActivitiyMaxInfo.MaxActivity > activitiyMaxInfo.MaxActivity)
            //                {
            //                    activitiyMaxInfo.MaxActivity = localActivitiyMaxInfo.MaxActivity;
            //                    activitiyMaxInfo.ActivityMax_MiniColumn = localActivitiyMaxInfo.ActivityMax_MiniColumn;
            //                }

            //                if (localActivitiyMaxInfo.MaxSuperActivity > activitiyMaxInfo.MaxSuperActivity)
            //                {
            //                    activitiyMaxInfo.MaxSuperActivity = localActivitiyMaxInfo.MaxSuperActivity;
            //                    activitiyMaxInfo.SuperActivityMax_MiniColumn = localActivitiyMaxInfo.SuperActivityMax_MiniColumn;
            //                }
            //            }
            //        });
        }

        private void SetColors_VisualizationTableItems(Cortex cortex)
        {
            Random random = new();

            VisualizationTableItemsCluster r_VisualizationTableItemsCluster = new()
            {
                Hash = new float[Constants.HashLength],
                TempHash = new float[Constants.HashLength],
                VisualizationTableItems = new(1000),
                MinCosineSimilarity = float.MaxValue,
                MaxCosineSimilarity = float.MinValue,
            };
            VisualizationTableItemsCluster g_VisualizationTableItemsCluster = new()
            {
                Hash = new float[Constants.HashLength],
                TempHash = new float[Constants.HashLength],
                VisualizationTableItems = new(1000),
                MinCosineSimilarity = float.MaxValue,
                MaxCosineSimilarity = float.MinValue,
            };
            VisualizationTableItemsCluster b_VisualizationTableItemsCluster = new()
            {
                Hash = new float[Constants.HashLength],
                TempHash = new float[Constants.HashLength],
                VisualizationTableItems = new(1000),
                MinCosineSimilarity = float.MaxValue,
                MaxCosineSimilarity = float.MinValue,
            };
            VisualizationTableItemsCluster[] clusters = [ r_VisualizationTableItemsCluster, g_VisualizationTableItemsCluster, b_VisualizationTableItemsCluster ];

            // Случайные начальные центры
            foreach (var cluster in clusters)
            {
                foreach (var _ in Enumerable.Range(0, Constants.InitialMemoryBitsCount))
                {
                    cluster.Hash[random.Next(Constants.HashLength)] = 1.0f;
                }
            }

            // Находим центры кластеров (EM)
            foreach (var _ in Enumerable.Range(0, 5))
            {
                foreach (var cluster in clusters)
                {
                    cluster.VisualizationTableItems.Clear();
                }
                
                foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
                {
                    var max = clusters.MaxBy(c => TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, c.Hash));
                    if (max is null)
                    {
                        throw new Exception();
                    }
                    else
                    {
                        max.VisualizationTableItems.Add(visualizationTableItem);
                    }
                }

                foreach (var cluster in clusters)
                {
                    Array.Clear(cluster.Hash);
                    if (cluster.VisualizationTableItems.Count == 0)
                        throw new Exception();
                    foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
                    {
                        TensorPrimitives.Add(cluster.Hash, visualizationTableItem.Hash, cluster.Hash);
                    }
                    TensorPrimitives.Divide(cluster.Hash, cluster.VisualizationTableItems.Count, cluster.Hash);
                }
            }

            //// Находим максимально удаленные точки от цетров кластеров
            //foreach (var cluster in clusters)
            //{
            //    var otherClusters = clusters.Where(c => !ReferenceEquals(c, cluster)).ToArray();
            //    float min = float.MaxValue;
            //    VisualizationTableItem? minVisualizationTableItem = null;
            //    foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
            //    {
            //        float d = 0.0f;
            //        foreach (var otherCluster in otherClusters)
            //        {
            //            d += TensorPrimitives.CosineSimilarity(otherCluster.Hash, visualizationTableItem.Hash);
            //        }
            //        if (d < min)
            //            minVisualizationTableItem = visualizationTableItem;
            //    }
            //    cluster.TempHash = minVisualizationTableItem!.Hash;
            //}

            //foreach (var cluster in clusters)
            //{
            //    Array.Copy(cluster.TempHash, cluster.Hash, cluster.TempHash.Length);
            //}

            foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
            {
                foreach (var cluster in clusters)
                {
                    var d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, cluster.Hash);
                    if (d < cluster.MinCosineSimilarity)
                        cluster.MinCosineSimilarity = d;
                    if (d > cluster.MaxCosineSimilarity)
                        cluster.MaxCosineSimilarity = d;
                }
            }
            foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
            {
                var r_d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, r_VisualizationTableItemsCluster.Hash);
                int r = 1 + (int)(254 * (r_d - r_VisualizationTableItemsCluster.MinCosineSimilarity) /
                    (r_VisualizationTableItemsCluster.MaxCosineSimilarity - r_VisualizationTableItemsCluster.MinCosineSimilarity));
                var g_d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, g_VisualizationTableItemsCluster.Hash);
                int g = 1 + (int)(254 * (g_d - g_VisualizationTableItemsCluster.MinCosineSimilarity) /
                    (g_VisualizationTableItemsCluster.MaxCosineSimilarity - g_VisualizationTableItemsCluster.MinCosineSimilarity));
                var b_d = TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, b_VisualizationTableItemsCluster.Hash);
                int b = 1 + (int)(254 * (b_d - b_VisualizationTableItemsCluster.MinCosineSimilarity) /
                    (b_VisualizationTableItemsCluster.MaxCosineSimilarity - b_VisualizationTableItemsCluster.MinCosineSimilarity));

                float k = 255.0f / Math.Max(r, Math.Max(g, b));
                if (float.IsInfinity(k) || float.IsNaN(k))
                    throw new Exception();
                if (r == 1 && g == 1 && b == 1)
                    throw new Exception();
                visualizationTableItem.Color = Color.FromArgb((int)(k * r), (int)(k * g), (int)(k * b));
            }
        }

        private void SetColors_VisualizationTableItems2(Cortex cortex)
        {
            Random random = new();            

            VisualizationTableItemsCluster[] clusters = new VisualizationTableItemsCluster[7];

            foreach (var ci in Enumerable.Range(0, clusters.Length))
            {
                clusters[ci] = new()
                {
                    Hash = new float[Constants.HashLength],
                    TempHash = new float[Constants.HashLength],
                    VisualizationTableItems = new(1000),
                    MinCosineSimilarity = float.MaxValue,
                    MaxCosineSimilarity = float.MinValue,
                };
            }

            // Случайные начальные центры
            foreach (var cluster in clusters)
            {
                foreach (var _ in Enumerable.Range(0, Constants.InitialMemoryBitsCount))
                {
                    cluster.Hash[random.Next(Constants.HashLength)] = 1.0f;
                }
            }

            // Находим центры кластеров (EM)
            foreach (var _ in Enumerable.Range(0, 5))
            {
                foreach (var cluster in clusters)
                {
                    cluster.VisualizationTableItems.Clear();
                }

                foreach (VisualizationTableItem visualizationTableItem in cortex.VisualizationTableItems)
                {
                    var max = clusters.MaxBy(c => TensorPrimitives.CosineSimilarity(visualizationTableItem.Hash, c.Hash));
                    if (max is null)
                    {
                        throw new Exception();
                    }
                    else
                    {
                        max.VisualizationTableItems.Add(visualizationTableItem);
                    }
                }

                foreach (var cluster in clusters)
                {
                    Array.Clear(cluster.Hash);
                    if (cluster.VisualizationTableItems.Count == 0)
                        throw new Exception();
                    foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
                    {
                        TensorPrimitives.Add(cluster.Hash, visualizationTableItem.Hash, cluster.Hash);
                    }
                    TensorPrimitives.Divide(cluster.Hash, cluster.VisualizationTableItems.Count, cluster.Hash);
                }
            }

            foreach (var ci in Enumerable.Range(0, clusters.Length))
            {
                var cluster = clusters[ci];
                foreach (VisualizationTableItem visualizationTableItem in cluster.VisualizationTableItems)
                {
                    visualizationTableItem.Color = DefaultColors[ci];
                }
            }
        }

        private Image GetMnistTotalBitmap()
        {
            GradientInPoint[,] totalGradientMatrix = new GradientInPoint[MNISTHelper.MNISTImageWidth, MNISTHelper.MNISTImageHeight];
            foreach (int i in Enumerable.Range(0, GradientMatricesCollection.Count))
            {
                GradientInPoint[,] gm = GradientMatricesCollection[i];
                for (int y = 1; y < MNISTHelper.MNISTImageHeight - 1; y += 1)
                {
                    for (int x = 1; x < MNISTHelper.MNISTImageWidth - 1; x += 1)
                    {
                        GradientInPoint p = gm[x, y];
                        GradientInPoint totalP = totalGradientMatrix[x, y];

                        totalP.GradX += p.GradX;
                        totalP.GradY += p.GradY;                        

                        totalGradientMatrix[x, y] = totalP;
                    }
                }                
            }
            //for (int y = 1; y < MNISTHelper.MNISTImageHeight - 1; y += 1)
            //{
            //    for (int x = 1; x < MNISTHelper.MNISTImageWidth - 1; x += 1)
            //    {                    
            //        GradientInPoint totalP = totalGradientMatrix[x, y];

            //        //totalP.GradX = totalP.GradX / GradientMatricesCollection.Count; // не надо делить, т.к. есть отрицательные значения
            //        //totalP.GradY = totalP.GradY / GradientMatricesCollection.Count; // не надо делить, т.к. есть отрицательные значения
            //        //totalP.Magnitude = totalP.Magnitude / GradientMatricesCollection.Count;
            //        //totalP.Angle = totalP.Angle / GradientMatricesCollection.Count; // не надо делить, т.к. есть отрицательные значения

            //        totalGradientMatrix[x, y] = totalP;
            //    }
            //}

            return Visualisation.GetGradientBigBitmap(totalGradientMatrix);
        }

        public static readonly Color[] DefaultColors =
        {
            Color.FromArgb(0xFF, 0x00, 0xFE),
            Color.FromArgb(0x02, 0x00, 0xF9),
            Color.FromArgb(0x00, 0xFF, 0xFF),
            Color.FromArgb(0xFF, 0x80, 0x41),
            Color.FromArgb(0xFC, 0x01, 0x00),
            Color.FromArgb(0x00, 0xFF, 0x01),
            Color.FromArgb(0xFF, 0xFF, 0x00),
            Color.FromArgb(0xFF, 0x00, 0x00),            
        };

        private class VisualizationTableItemsCluster
        {
            public float[] Hash = null!;

            public float[] TempHash = null!;

            public List<VisualizationTableItem> VisualizationTableItems = null!;

            public float MinCosineSimilarity;

            public float MaxCosineSimilarity;            
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
            public int? SubAreaMiniColumnsCount => null;

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
            public int MinBitsInHashForMemory => 8;

            /// <summary>
            ///     Максимальное расстояние до ближайших миниколонок
            /// </summary>
            public int NearestMiniColumnsDelta => 7;

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
        }        
    }
}
